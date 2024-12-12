from transformers import ClapModel
from typing import Any, List, Optional, Tuple, Union
import torch
from transformers.models.clap.modeling_clap import ClapOutput


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


class Model(ClapModel):
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            is_longer: Optional[torch.BoolTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            return_loss: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)

        text_embeds = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale_text = self.logit_scale_t.exp()
        logit_scale_audio = self.logit_scale_a.exp()
        logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio

        loss = None
        if return_loss:
            caption_loss = contrastive_loss(logits_per_text)
            audio_loss = contrastive_loss(logits_per_audio.t())
            loss = (caption_loss + audio_loss) / 2.0

        if not return_dict:
            output = (logits_per_audio, logits_per_text, text_embeds, audio_embeds, text_outputs, audio_outputs)
            return ((loss,) + output) if loss is not None else output

        return ClapOutput(
            loss=loss,
            logits_per_audio=logits_per_audio,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
        )
