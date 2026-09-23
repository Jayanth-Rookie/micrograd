"""
Demo and verification runner for micrograd.
"""
from micrograd.engine import Value
from micrograd.nn import MLP

def test_autograd_scalar():
    print("=" * 60)
    print("1. Testing Scalar Autograd Engine (Value)")
    print("=" * 60)
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    
    print(f"Forward output: g = {g.data:.6f}")
    print(f"Gradient dg/da: {a.grad:.6f}")
    print(f"Gradient dg/db: {b.grad:.6f}")
    
    # Expected known values
    expected_g = 24.704082
    expected_dg_da = 138.833819
    expected_dg_db = 645.577259
    assert abs(g.data - expected_g) < 1e-4, f"Mismatch in g: {g.data} vs {expected_g}"
    assert abs(a.grad - expected_dg_da) < 1e-4, f"Mismatch in dg/da: {a.grad} vs {expected_dg_da}"
    assert abs(b.grad - expected_dg_db) < 1e-4, f"Mismatch in dg/db: {b.grad} vs {expected_dg_db}"
    print("[PASS] Scalar autograd passed exact numerical gradient check!\n")

def test_mlp_training():
    print("=" * 60)
    print("2. Testing Neural Network (MLP) Training")
    print("=" * 60)
    # Initialize a 2-layer MLP: 3 inputs -> two hidden layers of 4 -> 1 output
    n = MLP(3, [4, 4, 1])
    print(f"Initialized MLP with {len(n.parameters())} parameters:")
    print(n)
    
    # Toy dataset: 4 examples, each with 3 input features
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    
    print("\nTraining MLP for 20 optimization steps:")
    learning_rate = 0.05
    initial_loss = None
    final_loss = None
    
    for k in range(20):
        # Forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        if k == 0:
            initial_loss = loss.data
            
        # Backward pass (zero grad then backward)
        n.zero_grad()
        loss.backward()
        
        # Gradient descent update
        for p in n.parameters():
            p.data -= learning_rate * p.grad
            
        if k % 5 == 0 or k == 19:
            preds_str = ", ".join([f"{yp.data:+.3f}" for yp in ypred])
            print(f"  Step {k:2d} | Loss: {loss.data:.6f} | Predictions: [{preds_str}]")
        final_loss = loss.data

    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during training!"
    print("[PASS] MLP training loop completed successfully with loss reduction!\n")

if __name__ == "__main__":
    test_autograd_scalar()
    test_mlp_training()
    print("All micrograd checks passed successfully!")
