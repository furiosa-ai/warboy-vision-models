import click

from .demo import *
from .make_model import *
from .performance_test import run_test


@click.group()
def cli():
    pass


cli.add_command(run_test.performance)

if __name__ == "__main__":
    cli()
