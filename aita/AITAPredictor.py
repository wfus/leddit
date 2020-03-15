from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('aita_predictor')
class AITAClassifier(Predictor):
    """Predictor wrapper for the AITA Transformer-based Classifier"""
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict['all_labels'] = all_labels
        output_dict['title'] = inputs['title']
        output_dict['selftext'] = inputs['selftext']
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict['title']
        post = json_dict['selftext']
        return self._dataset_reader.text_to_instance(
            title=title,
            post=post
        )
        
