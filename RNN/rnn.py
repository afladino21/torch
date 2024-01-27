import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, N_LETTERS
from utils import (load_data, letter_to_tensor,
                   line_to_tensor, random_training_example)


class RNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        # ? hidden_size: Tamaño del vector del vector de estados
        self.hidden_dize = hidden_size
        # ? input_size = tamaño vector de entrada
        # ? output_size: Como es un problema de clasificación serían las clases
        # Input to hidden
        self.i2h = nn.Linear(in_features=input_size +
                             hidden_size, out_features=hidden_size)
        # Input to output
        self.i2o = nn.Linear(in_features=input_size +
                             hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)  # [1,57] second dim

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dize)


def category_from_index(output, categories):
    categorie_id = torch.argmax(output)
    return categories[categorie_id]


def train(rnn, line_tensor, category_tensor, criterion, optimizer):
    hidden = rnn.init_hidden()

    for idx_tensor in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[idx_tensor], hidden)  # Forward pass

    loss = criterion(output, category_tensor)
    # optimizer.zero_grad(): Acá tambien puede ir
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, loss.item()


def predict(rnn, input_line, category_from_output, categories):
    print(f"\n> {input_line}")
    # with torch.no_grad():
    line_tensor = line_to_tensor(input_line)
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    guess = category_from_output(output.detach(), categories)

    return guess


def main():
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)

    HIDDEN_SIZE = 128

    rnn = RNN(input_size=N_LETTERS, hidden_size=HIDDEN_SIZE,
              output_size=n_categories)

    # ONE STEP
    if False:
        input_tensor = letter_to_tensor('a')
        hidden_tensor = rnn.init_hidden()
        output, next_hidden = rnn(input_tensor, hidden_tensor)
        print(output)

    # WHOLE SEQUENCE
    if False:
        input_tensor = line_to_tensor('Andres')
        hidden_tensor = rnn.init_hidden()
        output, next_hidden = rnn(input_tensor[0], hidden_tensor)

    criterion = nn.NLLLoss()
    learning_rate = 5e-3
    optimzer = torch.optim.SGD(params=rnn.parameters(), lr=learning_rate)

    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    n_iters = 100000

    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example(
            category_lines, all_categories)

        output, loss = train(rnn=rnn, line_tensor=line_tensor, category_tensor=category_tensor,
                             criterion=criterion, optimizer=optimzer)

        current_loss += loss

        if (i+1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        if (i+1) % print_steps == 0:
            guess = category_from_index(output, all_categories)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

    plt.figure()
    plt.plot(all_losses)
    plt.show()

    while True:
        sentence = input("Input:")
        if sentence == "quit":
            break

        prediction = predict(
            rnn, sentence, category_from_index, all_categories)
        print(prediction)


if __name__ == '__main__':
    main()
