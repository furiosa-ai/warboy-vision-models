import click

from .demo import run_demo
from .make_model import run_make_model
from .performance_test import run_test_scenarios


@click.group()
def cli():
    pass


cli.add_command(run_test_scenarios.run_e2e_test)
cli.add_command(run_demo.run_demo)
cli.add_command(run_make_model.run_make_model)
cli.add_command(run_test_scenarios.run_npu_performance_test)

if __name__ == "__main__":
    cli()
