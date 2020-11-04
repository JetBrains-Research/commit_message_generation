class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        :param x: [batch_size, target_sequence_length, decoder_hidden_size]
        :param y: [batch_size, target_sequence_length]
        :return: float
        """
        x = self.generator(x)  # [batch_size, target_sequence_length, vocab_size]
        # x.contiguous().view(-1, x.size(-1) - [batch_size * target_sequence_length, vocab_size]
        # y.contiguous().view(-1) - [batch_size * target_sequence_length]

        loss = self.criterion(x.view(-1, x.size(-1)),
                              y.view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm
