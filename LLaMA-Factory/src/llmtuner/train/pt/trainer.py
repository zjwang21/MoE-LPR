from typing import Dict
import torch
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        balance_loss = outputs.balance_loss
        lpr_loss = outputs.lpr_loss

        if balance_loss != None:
            nlayers = len(outputs.router_logits)
            prefix = "load_balance_"
            with torch.no_grad():
                mask = inputs['attention_mask'].reshape(-1) # n
                mask = mask.unsqueeze(0).expand(nlayers, mask.size(0)).bool() # nlayers x n
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # nlayers x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1) # nlayers x n x e
                probs = torch.mean(probs[mask], dim=0).detach().cpu() # e

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = balance_loss.item()
                logs["scores_per_expert"] = " ".join([str(round(k, 2)) for k in probs.tolist()])
                self.log(logs)

        if lpr_loss != None:
            prefix = "lpr_"
            lang_mask = inputs['langs']
            with torch.no_grad():
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # nlayers x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1)
                mask = lang_mask.reshape(-1).bool().expand(probs.size()[:2])
                probs = probs[mask].to(torch.float).reshape(mask.size(0), -1, probs.size(-1)) # nlayers x n x e
                probs = probs[:, :, 0] # nlayers x n
                score_expert0 = torch.mean(probs, dim=-1).detach().cpu()

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = lpr_loss.item()
                logs["old_lang_expert0_score"] = " ".join([str(round(k, 2)) for k in score_expert0.tolist()])
                self.log(logs)

        if self.main_loss_logged: self.main_loss_logged = False
        return loss