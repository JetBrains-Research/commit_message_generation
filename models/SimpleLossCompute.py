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
        :param norm: normalizing coefficient (usually batch size)
        :return: float
        """
        x = self.generator(x)  # [batch_size, target_sequence_length, vocab_size]
        print("Loss")
        print(x.contiguous().view(-1, x.size(-1)))
        print(y.contiguous().view(-1))

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm
