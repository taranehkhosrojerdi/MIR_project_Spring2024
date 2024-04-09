
from typing import List
import math
import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        # TODO: Calculate precision here
        true_positives = 0
        total_predicted_positives = 0

        for i in range(len(actual)):
            actual_set = set(actual[i])
            predicted_set = set(predicted[i])
            true_positives += len(actual_set.intersection(predicted_set))
            total_predicted_positives += len(predicted_set)
        precision = true_positives / total_predicted_positives if total_predicted_positives!= 0 else 0.0
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        # TODO: Calculate recall here
        true_positives = 0
        total_actual_positives = 0

        for i in range(len(actual)):
            actual_set = set(actual[i])
            predicted_set = set(predicted[i])
            true_positives += len(actual_set.intersection(predicted_set))
            total_actual_positives += len(actual_set)
        recall = true_positives / total_actual_positives if total_actual_positives!= 0 else 0.0
        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        # TODO: Calculate F1 here
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        # TODO: Calculate AP here
        for i in range(len(actual)):
            num_hits = 0
            precision_sum = 0.0
            for j, item in enumerate(predicted[i]):
                if item in actual[i]:
                    num_hits += 1
                    precision_sum += num_hits / (j + 1)
            if num_hits > 0:
                AP += precision_sum / num_hits

        AP /= len(actual)
        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        # TODO: Calculate MAP here
        num_queries = len(actual)
        for i in range(num_queries):
            num_hits = 0
            precision_sum = 0.0
            for j, item in enumerate(predicted[i]):
                if item in actual[i]:
                    num_hits += 1
                    precision_sum += num_hits / (j + 1)
            if num_hits > 0:
                MAP += precision_sum / num_hits
        MAP /= num_queries
        return MAP
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        # TODO: Calculate DCG here
        for i in range(len(actual)):
            DCG_i = 0.0
            for j in range(len(predicted[i])):
                relevance = 0
                if predicted[i][j] in actual[i]:
                    relevance = 1
                DCG_i += (2 ** relevance - 1) / math.log2(j + 2)
            DCG += DCG_i
        return DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        # TODO: Calculate NDCG here

        return NDCG
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        # TODO: Calculate MRR here
        for i in range(len(actual)):
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    RR += 1 / (j + 1)
                    break
        RR /= len(actual)
        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        # TODO: Calculate MRR here

        return MRR
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        #TODO: Print the evaluation metrics
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        print(f"Discounted Cumulative Gain: {dcg}")
        print(f"Normalized Discounted Cumulative Gain: {ndcg}")
        print(f"Reciprocal Rank: {rr}")
        print(f"Mean Reciprocal Rank: {mrr}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        #TODO: Log the evaluation metrics using Wandb
        wandb.init(project="evaluation_metrics", name=self.name)
    
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map,
            "Discounted Cumulative Gain": dcg,
            "Normalized Discounted Cumulative Gain": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)


# -------------------------------------------Test--------------------------------------------

