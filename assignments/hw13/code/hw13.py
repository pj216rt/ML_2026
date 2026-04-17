import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

#need to load image and convert to array
image = Image.open("assignments/hw13/data/gw.bmp")
img_array = np.array(image)

#part a

#substract the minimum of each column
column_minimums = np.min(img_array, axis=0)
image_norm = img_array - column_minimums

#display the normalized image
plt.figure(figsize=(8, 4))
plt.imshow(image_norm, cmap="gray")
plt.title("Normalized guidewire image")
plt.tight_layout()
plt.axis("off")
plt.savefig("assignments/hw13/figures/normalized_image.pdf", dpi=400, bbox_inches="tight")
#plt.show()
plt.close()

#need a plot with a curve through the minima
#need to the get the locations of the minimum values from each column
min_locations = np.argmin(img_array, axis=0)
cols = np.arange(img_array.shape[1])
rows = min_locations

plt.figure(figsize=(8, 4))
plt.imshow(image_norm, cmap="gray")
plt.plot(cols, rows, color="red")
plt.title("Normalized guidewire image with minimum-value location curve")
plt.tight_layout()
plt.savefig("assignments/hw13/figures/gw_with_curve.pdf", dpi=400, bbox_inches="tight")
#plt.show()
plt.close()


#parts
results = {}
alphas = [2, 15]

#get dimensions of normalized image
rows, columns = image_norm.shape

#rows are states here
states = np.arange(rows)

#I(i,j) is the pixel intensity at row i and column j of the image
#Viterbi algorithm, similar to what we did in the HMM homework
#difference is we are trying to minimize, whereas Viterbi is a maximization problem
#https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
for alpha in alphas:
    
    #minimum cost of path ending in state i column j
    prob = np.full((columns, rows), np.inf)

    #previous state in the optimal path to state i column j
    prev = np.zeros((columns, rows), dtype=int)

    #initialize first column
    for s in states:
        prob[0, s] = image_norm[s, 0]

    #now need the recursion step
    for column in range(1, columns):
        for s in states:
            for r in states:
                new_cost = prob[column - 1, r] + alpha * (s - r)**2 + image_norm[s, column]

                if new_cost < prob[column, s]:
                    prob[column, s] = new_cost
                    prev[column, s] = r

    path = np.zeros(columns, dtype=int)
    path[columns - 1] = np.argmin(prob[columns - 1, :]) 

    #go backwards now
    for t in range(columns - 2, -1, -1):
        path[t] = prev[t + 1, path[t + 1]]

    #minimum value
    f_min = prob[columns - 1, path[columns - 1]]

    #store results
    results[alpha] = {
        "f_min": f_min,
        "curve": path
    }

#create dataframe to save reporting to
df = pd.DataFrame({
    "alpha": [2, 15],
    "f_min": [results[2]["f_min"], results[15]["f_min"]]
})
df.columns = [r"$\alpha$", r"Minimum $f(c)$"]

#save table to latex
df.to_latex(
    "assignments/hw13/output/results_table.tex",
    index=False,
    float_format="%.4f",
    escape=False
)

#plotting is pretty easy now
colors = {2: "red", 15: "blue"}

for alpha in [2, 15]:
    plt.figure(figsize=(8, 4))
    plt.imshow(image_norm, cmap="gray")
    plt.plot(np.arange(image_norm.shape[1]), results[alpha]["curve"], color=colors[alpha], linewidth=2)
    plt.title(fr"Normalized image with optimal curve, $\alpha = {alpha}$")
    plt.tight_layout()
    plt.savefig(f"assignments/hw13/figures/gw_curve_alpha_{alpha}.pdf",
                dpi=400, bbox_inches="tight")
    #plt.show()
    plt.close()

    #images are really wide and long.  Gotta be a way to change this shape