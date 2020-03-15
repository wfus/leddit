from aita.AITAPredictor import AITAClassifier
from aita.test_reader import AITATestReader
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive 

if __name__ == '__main__':
    archive = load_archive('/tmp/roberta-base.tar.gz')
    predictor = AITAClassifier.from_archive(archive)
    validation_instances = predictor._dataset_reader.read('./data/aita-tiny-test.pkl')
    gradients, outputs = predictor.get_gradients(validation_instances)
    print(gradients)