from trainers.BaseTrainer import *

class AdvTrainer(BaseTrainer):
    def __init__(self, model_name, config, vectors, **kwargs):
        super(AdvTrainer, self).__init__(model_name, config, vectors, **kwargs)

    def get_optimizer(self, lr, optimizer, weight_decay=1e-3, use_amp=False):
        # optimizers = self.cur_model.get_optimizer(lr, optimizer, weight_decay, autocast)
        optimizers = super().get_optimizer(lr, optimizer, weight_decay, use_amp)
        self.dis_optimizers = self.cur_model.get_dis_optimizer(self.model, lr, optimizer, weight_decay, use_amp)
        return optimizers

    def dis_optimizer_step(self):
        # self.optimizer.step()
        for optimizer in self.dis_optimizers:
            optimizer.step()

    def dis_optimizer_zero_grad(self):
        # self.optimizer.zero_grad()
        for optimizer in self.dis_optimizers:
            optimizer.zero_grad()
    def batch_process(self, x, values, lengths, masks, ids, times, cur_graphs, dataloader, aux):
        # self.optimizer_zero_grad()
        if self.model.training:
            predicted_values, aux_output, other_output = self.model(x, lengths, masks, ids, cur_graphs, times)
        else:
            predicted_values, aux_output, other_output = self.cur_model.predict(x, lengths, masks, ids, cur_graphs, times)
        # aux
        loss = self.get_reg_loss_and_result(values, predicted_values, self.criterion,
                                            log=dataloader.log,
                                            mean=dataloader.mean, std=dataloader.std)
        if aux:
            aux_loss = self.get_aux_loss_and_result(values, aux_output)
            loss = loss + aux_loss

        if other_output['loss'] is not None:
            loss = loss + other_output['loss']

        if self.model.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer_step()
            self.optimizer_zero_grad()
            if self.epoch >= self.cur_model.dis_start:
                for i in range(self.cur_model.dis_epoch):
                    dis_loss = self.cur_model.get_dis_loss(other_output['input'], phase='dis_update')
                    dis_loss.backward()
                    self.dis_optimizer_step()
                    self.dis_optimizer_zero_grad()
            self.cur_model.train()

        return loss
