import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd


class ABTestManager:
    def __init__(self, recommendation_algorithms: Dict[str, callable]):
        self.algorithms = recommendation_algorithms
        self.active_tests = {}
        self.results = pd.DataFrame(columns=['test_id', 'algorithm_a', 'algorithm_b',
                                             'metric', 'value', 'p_value', 'significant'])

    def start_test(self, test_id: str, algorithm_a: str, algorithm_b: str,
                   metric: str = 'conversion_rate', sample_size: int = 1000) -> None:
        self.active_tests[test_id] = {
            'algorithm_a': algorithm_a,
            'algorithm_b': algorithm_b,
            'metric': metric,
            'group_a': {'samples': [], 'size': sample_size},
            'group_b': {'samples': [], 'size': sample_size},
            'completed': False
        }

    def log_interaction(self, test_id: str, group: str, success: bool) -> None:
        if test_id not in self.active_tests:
            return

        test = self.active_tests[test_id]
        if group not in ['a', 'b']:
            return

        group_key = f'group_{group}'
        test[group_key]['samples'].append(1 if success else 0)

        # Check if test is complete
        if (len(test['group_a']['samples']) >= test['group_a']['size'] and
                len(test['group_b']['samples']) >= test['group_b']['size']):
            self._complete_test(test_id)

    def _complete_test(self, test_id: str) -> None:
        test = self.active_tests[test_id]

        # Calculate metric values
        metric_a = np.mean(test['group_a']['samples'])
        metric_b = np.mean(test['group_b']['samples'])

        # Calculate statistical significance
        t_stat, p_value = stats.ttest_ind(
            test['group_a']['samples'],
            test['group_b']['samples'],
            equal_var=False
        )

        # Store results
        self.results = self.results.append({
            'test_id': test_id,
            'algorithm_a': test['algorithm_a'],
            'algorithm_b': test['algorithm_b'],
            'metric': test['metric'],
            'value_a': metric_a,
            'value_b': metric_b,
            'p_value': p_value,
            'significant': p_value < 0.05
        }, ignore_index=True)

        test['completed'] = True

    def get_recommendation(self, test_id: str, student_id: int) -> Tuple[str, List[int]]:
        if test_id not in self.active_tests:
            return 'default', self.algorithms['default'](student_id)

        test = self.active_tests[test_id]
        if test['completed']:
            return 'default', self.algorithms['default'](student_id)

        # Randomly assign to group A or B
        group = 'a' if np.random.random() < 0.5 else 'b'
        algorithm = test[f'algorithm_{group}']

        return group, self.algorithms[algorithm](student_id)

    def get_test_results(self, test_id: str) -> Dict:
        if test_id not in self.active_tests:
            return None

        return self.results[self.results['test_id'] == test_id].to_dict('records')