#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import umap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb_csv",
        default="/mnt/volumec/Aswath/patchmodel/s3cima/CNNResnetembeddings/arcsin_256_512d_linearadapter.csv"
    )
    parser.add_argument(
        "--out_png",
        default="cnn_embedding_umaplinear.png"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.emb_csv)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    y = df["Tissue"].values

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        n_components=2,
        metric="cosine",
        random_state=42
    )

    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(7, 7))
    colors = {
        "core": "red",
        "rim": "orange",
        "normalLiver": "green"
    }

    for t in ["core", "rim", "normalLiver"]:
        m = (y == t)
        plt.scatter(
            X2[m, 0],
            X2[m, 1],
            s=2,
            alpha=0.6,
            label=t,
            c=colors[t]
        )

    plt.legend(markerscale=4)
    plt.title("CNN Embedding Separability (UMAP)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()

    print(f"✅ UMAP saved to {args.out_png}")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--emb_csv",
#         default="/mnt/volumec/Aswath/patchmodel/s3cima/CNN3layerMarker/fast_256_128d.csv"
#     )
#     parser.add_argument(
#         "--out_png",
#         default="cnn_embedding_pca.png"
#     )
#     args = parser.parse_args()

#     # ---------------- Load embeddings ----------------
#     df = pd.read_csv(args.emb_csv)

#     emb_cols = [c for c in df.columns if c.startswith("emb_")]
#     assert len(emb_cols) > 0, "No embedding columns found!"

#     X = df[emb_cols].values
#     y = df["Tissue"].values

#     # ---------------- PCA ----------------
#     X2 = PCA(n_components=2, random_state=42).fit_transform(X)

#     # ---------------- Plot ----------------
#     plt.figure(figsize=(7, 7))

#     for t, c in zip(
#         ["core", "rim", "normalLiver"],
#         ["red", "orange", "green"]
#     ):
#         m = (y == t)
#         plt.scatter(
#             X2[m, 0],
#             X2[m, 1],
#             s=2,
#             alpha=0.6,
#             label=t,
#             c=c
#         )

#     plt.legend(markerscale=4)
#     plt.title("CNN Embedding Separability (PCA)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.tight_layout()

#     # ---------------- Save ----------------
#     plt.savefig(args.out_png, dpi=300)
#     plt.close()

#     print(f" PCA plot saved to: {args.out_png}")

# if __name__ == "__main__":
#     main()