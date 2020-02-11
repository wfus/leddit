from AITAReader import AITASimpleOnelineDataset
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    reader = AITASimpleOnelineDataset(resample_labels=True)
    reader.read("aita-train.pkl")