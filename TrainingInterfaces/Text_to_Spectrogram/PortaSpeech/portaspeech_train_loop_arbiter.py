from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_meta_train_loop import (
    train_loop as multi_language_loop,
)
from TrainingInterfaces.Text_to_Spectrogram.PortaSpeech.portaspeech_train_loop import (
    train_loop as mono_language_loop,
)


def train_loop(
    net,  # an already initialized PortaSpeech model that should be trained.
    datasets,
    # a list of datasets to train on. Every dataset within a language should already be a concat dataset of all the datasets
    # in that language. So every list entry here should be a (combined) dataset for each language. For the case of a monolingual model, pass a list
    # with only one dataset in it. This will trigger the arbiter to call the train loop for simple one language training runs rather than the complex
    # LAML based one.
    device,  # the device where this training should run on.
    save_directory,  # directory where the models and visualizations should be saved.
    steps_per_checkpoint=1000,
    # how many steps should be trained before a checkpoint is created. This is only relevant for the multilingual case,
    # the monolingual case will do this once per epoch, regardless of the steps.
    path_to_checkpoint=None,  # path to a trained checkpoint to either continue training or fine-tune from.
    lr=0.001,  # learning rate of the model.
    path_to_embed_model="Models/Embedding/embedding_function.pt",
    # path to the utterance embedding function that is to be used.
    resume=False,  # whether to automatically load the most recent checkpoint and resume training from it.
    warmup_steps=8000,
    # how many steps until the learning rate reaches the specified value and starts decreasing again.
    use_wandb=False,
    # whether to use online experiment tracking with weights and biases. Requires prior CLI login.
    batch_size=64,
    # how many samples to put into one batch. Higher batch size is more stable, but requires more VRAM.
    eval_lang="en",  # in which language the evaluation sentence is to be plotted.
    fine_tune=False,  # whether to use the provided checkpoint as basis for fine-tuning.
    phase_1_steps=80000,  # without cycle consistency objective.
    phase_2_steps=120000,  # with cycle consistency objective.
):
    if type(datasets) != list:
        datasets = [datasets]
    if len(datasets) > 1:
        multi_language_loop(
            net=net,
            datasets=datasets,
            device=device,
            save_directory=save_directory,
            batch_size=batch_size,
            phase_1_steps=phase_1_steps,
            phase_2_steps=phase_2_steps,
            steps_per_checkpoint=steps_per_checkpoint,
            lr=lr,
            lang=eval_lang,
            path_to_checkpoint=path_to_checkpoint,
            path_to_embed_model=path_to_embed_model,
            resume=resume,
            fine_tune=fine_tune,
            warmup_steps=warmup_steps,
            use_wandb=use_wandb,
        )
    else:
        mono_language_loop(
            net=net,
            train_dataset=datasets[0],
            device=device,
            save_directory=save_directory,
            batch_size=batch_size,
            lang=eval_lang,
            lr=lr,
            warmup_steps=warmup_steps,
            path_to_checkpoint=path_to_checkpoint,
            path_to_embed_model=path_to_embed_model,
            fine_tune=fine_tune,
            resume=resume,
            phase_1_steps=phase_1_steps,
            phase_2_steps=phase_2_steps,
            use_wandb=use_wandb,
        )
