# 🧠 My First AI Brain

This is my first neural network — fully coded **from scratch** in Python using only NumPy.  
No frameworks, no magic. I built this to really understand how an AI learns.

## ⚙️ What it does
A simple feed-forward network that learns to solve the **XOR logic problem**.  
It starts out guessing randomly, then teaches itself the correct pattern through  
**forward propagation**, **backpropagation**, and **gradient descent**.

Input → Hidden (4 neurons) → Output (1 neuron)

After training, it predicts:

[0, 1, 1, 0] ✅ (100% accuracy)

## 🧩 How it works
1. **Initialization** — Xavier weights + tiny random biases for stability.  
2. **Activation** — Sigmoid neurons turn raw numbers into smooth probabilities.  
3. **Loss** — Binary Cross-Entropy measures how wrong the model is.  
4. **Backpropagation** — calculates how each weight affected the error.  
5. **Gradient descent** — updates the weights to make fewer mistakes each round.

---

## 🔧 Run it yourself
pip install numpy
python network.py


Output shows the training progress and final accuracy.

🎓 What I learned

How neurons, activations, weights, and biases actually work

How learning = minimizing error step by step

How to build AI logic without using TensorFlow or PyTorch

🏗️ Next step

Next up: scaling this idea to recognize handwritten digits (multi-class problem)
in the file digits_softmax.py.








© 2025 Kevin Mast – Built as part of my AI learning journey.
