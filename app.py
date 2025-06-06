import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Análise de Dobras em Tubo 3D", layout="centered")
st.title("🌐 Análise de Dobra de Tubos 3D")
st.markdown("Envie um arquivo CSV contendo colunas **X, Y, Z** com as coordenadas do tubo.")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not set(['X', 'Y', 'Z']).issubset(df.columns):
            st.error("O arquivo precisa conter as colunas: X, Y, Z")
        else:
            points = df[['X', 'Y', 'Z']].to_numpy()

            def vector_angle(v1, v2):
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                return np.degrees(np.arccos(cos_theta))

            results = []
            for i in range(1, len(points) - 1):
                p0, p1, p2 = points[i - 1], points[i], points[i + 1]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = vector_angle(v1, v2)
                normal = np.cross(v1, v2)
                ref_plane = np.array([0, 0, 1])
                giro = vector_angle(ref_plane, normal) if np.linalg.norm(normal) > 1e-6 else 0
                advance = np.linalg.norm(v1)

                results.append({
                    "Etapa": i,
                    "Avanço (mm)": round(advance, 2),
                    "Ângulo de Dobra (°)": round(angle, 2),
                    "Giro (°)": round(giro, 2)
                })

            df_result = pd.DataFrame(results)
            st.success("Arquivo processado com sucesso!")
            st.subheader("Resultados das Dobras")
            st.dataframe(df_result, use_container_width=True)

            st.subheader("Visualização 3D da Trajetória do Tubo")
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')
            X, Y, Z = df["X"], df["Y"], df["Z"]
            ax.plot(X, Y, Z, '-o', color='blue')
            for i, (x, y, z) in enumerate(zip(X, Y, Z)):
                ax.text(x, y, z + 5, f'P{i}', fontsize=8)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.view_init(elev=20., azim=-60)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Envie um arquivo CSV para iniciar.")
