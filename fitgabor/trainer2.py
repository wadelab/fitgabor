import numpy as np
import torch
from torch import optim
from tqdm import trange


def trainer_fn2(dog_gen, model_neuron,
               epochs=20000, lr=5e-3,
               fixed_std=.01,
               save_rf_every_n_epoch=None,
               optimizer=optim.Adam):
    dog_generator.apply_changes()

    optimizer = optimizer(dog_generator.parameters(), lr=lr)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    old_lr = lr
    lr_change_counter = 0

    pbar = trange(epochs, desc="Loss: {}".format(np.nan), leave=True)
    saved_rfs = []
    for epoch in pbar:
        
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if old_lr != current_lr:
            old_lr = current_lr
            lr_change_counter += 1

        if lr_change_counter > 3:
            break

        def closure():
            optimizer.zero_grad()

            # generate dog
            dog = dog_generator()

            if fixed_std is not None:
                dog_std = dog.std()
                dog_std_constrained = fixed_std * dog / dog_std

            loss = -model_neuron(dog_std_constrained)

            loss.backward()

            return loss

        loss = optimizer.step(closure)

        pbar.set_description("Loss: {:.2f}".format(loss.item()))

        if save_rf_every_n_epoch is not None:
            dog = dog_generator()
            if (epoch % save_rf_every_n_epoch) == 0:
                saved_rfs.append(dog.squeeze().cpu().data.numpy())

        lr_scheduler.step(-loss)

    dog_generator.eval();
    return dog_generator, saved_rfs