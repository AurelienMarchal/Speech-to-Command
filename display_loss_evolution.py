import matplotlib.pyplot as plt


train_log_file = 'results/enc_dec/1998/train_log.txt'


epochs = []
train_losses = []
valid_losses = []

with open(train_log_file, 'r') as f:
        for line in f.read().splitlines():
            epoch_str = line.split(',')[0]
            if "epoch:" in epoch_str:
                epoch = int(epoch_str.split(' ')[1])
                epochs.append(epoch)

                losses_str = line.split(',')[1]

                train_loss_str = losses_str.split(' - ')[1]
                valid_loss_str = losses_str.split(' - ')[2]

                train_loss = float(train_loss_str.split(': ')[1])
                valid_loss = float(valid_loss_str.split(': ')[1])

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

fig = plt.figure()

ax_lin = fig.add_subplot(1, 1, 1)
line, = ax_lin.plot(epochs, train_losses, color='blue', lw=2, label='train loss')
line, = ax_lin.plot(epochs, valid_losses, color='red', linestyle='dashed',lw=2, label='valid loss')
plt.xticks(epochs)
plt.legend(loc='best')

plt.savefig('./graphs/loss_evolution_without_data_augment.png')