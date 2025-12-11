
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from models.ae_policy import KFACOptimizer

class TestKFAC(unittest.TestCase):
    def test_kfac_step(self):
        """
        Verify KFAC optimizer runs without error and updates parameters.
        """
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        optimizer = KFACOptimizer(model, lr=0.01)

        input = torch.randn(4, 10)
        target = torch.randn(4, 1)

        # Forward
        output = model(input)
        loss = nn.MSELoss()(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check params before
        param_before = model[0].weight.clone()

        # Step
        optimizer.step()

        # Check params after
        param_after = model[0].weight
        self.assertFalse(torch.equal(param_before, param_after), "Parameters should update")

    def test_compare_vs_sgd(self):
        """
        Compare KFAC vs SGD on a simple regression task and log results.
        """
        print("\n--- Comparing KFAC vs SGD ---")

        def train_model(opt_class, name):
            torch.manual_seed(42)
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.Tanh(),
                nn.Linear(20, 1)
            )
            if name == "KFAC":
                optimizer = opt_class(model, lr=0.01)
            else:
                optimizer = opt_class(model.parameters(), lr=0.01)

            loss_fn = nn.MSELoss()
            input = torch.randn(100, 10)
            target = torch.randn(100, 1)

            losses = []
            for epoch in range(20):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            return losses

        sgd_losses = train_model(optim.SGD, "SGD")
        kfac_losses = train_model(KFACOptimizer, "KFAC")

        print(f"{'Epoch':<5} | {'SGD Loss':<10} | {'KFAC Loss':<10}")
        for i, (l_sgd, l_kfac) in enumerate(zip(sgd_losses, kfac_losses)):
            print(f"{i:<5} | {l_sgd:<10.4f} | {l_kfac:<10.4f}")

        # Basic assertion that KFAC decreases loss
        self.assertLess(kfac_losses[-1], kfac_losses[0])
        print("Comparison complete.")

if __name__ == '__main__':
    unittest.main()
