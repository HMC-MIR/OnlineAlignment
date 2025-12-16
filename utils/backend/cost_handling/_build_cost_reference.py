"""Build dictionary for checking cost metrics"""

# standard imports
import pickle

# custom imports
from core.cost import CostMetric, CosineDistance, EuclideanDistance


def build_cost_metric_dict():
    """Builds cost metric dictionary for user input.

    Note: This function is deprecated. The registry is now maintained directly
    in cost_handling.py. This is kept for reference only.
    """
    cost_metric_dict: dict[str, CostMetric] = {}
    cost_metric_dict["cosine"] = CosineDistance()
    cost_metric_dict["euclidean"] = EuclideanDistance()
    return cost_metric_dict


if __name__ == "__main__":
    cost_metric_dict = build_cost_metric_dict()
    with open("./supported_costs.pkl", "wb") as f:
        pickle.dump(cost_metric_dict, f)
