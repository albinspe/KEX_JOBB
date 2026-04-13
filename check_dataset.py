
from datasets import load_from_disk
import matplotlib.pyplot as plt

chunk_path = "/Users/albinspeijer/Documents/Programmering/Data_to_S3/openfake_arrow_chunks_50_50/chunk_0000"
ds = load_from_disk(chunk_path)

print(ds)
print(ds[0].keys())

n = 9
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ex = ds[i]
    ax.imshow(ex["image"])
    ax.set_title(f'{i} | {ex.get("label","?")}')
    ax.axis("off")

plt.tight_layout()
plt.show()

