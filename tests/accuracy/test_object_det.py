

class MSCOCODataLoader:
    """Data loader for MSCOCO dataset"""

    def __init__(
        self, image_dir: Path, annotations_file: Path, preprocess: Callable, input_shape
    ) -> None:
        self.coco = COCO(annotations_file)
        coco_images = self.coco.dataset["images"]
        self.image_paths = list(image_dir / image["file_name"] for image in coco_images)
        self.image_filename_to_annotation = {
            image["file_name"]: image for image in coco_images
        }
        self.preprocess = preprocess
        self.input_shape = input_shape

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict]]:
        for path in self.image_paths:
            img = cv2.imread(str(path))
            img0shape = img.shape[:2]
            yield self.preprocess(
                img, new_shape=self.input_shape
            ), self.image_filename_to_annotation[path.name], img0shape

    def __len__(self) -> int:
        return len(self.image_paths)


def warboy_yolo_accuracy_test(
    task: str,
):
    CONF_THRES = 0.001
    IOU_THRES = 0.7

    preprocessor = YOLOPreProcessor()
    