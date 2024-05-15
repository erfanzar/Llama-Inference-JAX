from src.llama_interface_jax.covertors import convert_llama_model
import pickle as pkl

if __name__ == "__main__":
    lijax_model = convert_llama_model("erfanzar/LinguaMatic-Tiny")
    pkl.dump(lijax_model, open("lijax_llama_model.pkl", "wb"))
