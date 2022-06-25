import torch
import pytorch_lightning as pl
import pathlib
import typing as tp
import ptrnets
import ptrnets.metrics as metrics
import click


@click.command()
@click.argument("experiment-dir")
@click.option(
    "--test-npoints",
    type=click.Choice(tp.get_args(ptrnets.ConvexHull.NPointsT)),
    required=True,
    multiple=True,
)
@click.option("--batch-size", default=128)
def main(experiment_dir, test_npoints, batch_size):
    print(experiment_dir, test_npoints, batch_size)
    ckpt_path = str(next((pathlib.Path(experiment_dir) / "checkpoints").iterdir()))
    print(ckpt_path)
    model = ptrnets.PointerNetworkForConvexHull.load_from_checkpoint(ckpt_path)
    datamodule = ptrnets.ConvexHullDataModule("data", "50", test_npoints, batch_size)
    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else 0)
    trainer = pl.Trainer()
    trainer.test(model, datamodule)

    # datamodule.setup("test")
    # inputs, decoder_inputs, targets = next(iter(datamodule.test_dataloader()[0]))
    # res = model.batch_beam_search(inputs)
    # breakpoint()
    # pass


@click.command()
def dummy_main():
    pad = torch.nn.utils.rnn.pad_packed_sequence
    pack = torch.nn.utils.rnn.pack_sequence
    model = ptrnets.PointerNetworkForConvexHull.load_from_checkpoint(
        "relela-02-logs/convex-hull/version_1/checkpoints/epoch=110-val_sequence_acc=0.807.ckpt"
    )
    pl.seed_everything(42)
    seqs = [torch.rand(5, 2), torch.rand(7, 2), torch.rand(10, 2), torch.rand(15, 2)]

    scores = model(pack(seqs, enforce_sorted=False), pack(seqs, enforce_sorted=False))

    inputs = torch.nn.utils.rnn.pack_sequence(
        seqs,
        enforce_sorted=False,
    )


    bad_input = torch.cat([seqs[1], torch.zeros(8, 2)])

    breakpoint()


if __name__ == "__main__":
    dummy_main()
