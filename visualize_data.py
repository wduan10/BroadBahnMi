import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

labels_path = '/groups/CS156b/data/student_labels/train2023.csv'
df = pd.read_csv(labels_path)
print(df)