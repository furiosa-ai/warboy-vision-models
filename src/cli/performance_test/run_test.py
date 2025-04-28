import click

from ...test_scenarios.e2e import object_det


@click.command()
def performance():
    object_det.test_warboy_yolo_accuracy_det(
        "test_config.yaml",
        "datasets/coco/val2017",
        "datasets/coco/annotations/instances_val2017.json",
    )
