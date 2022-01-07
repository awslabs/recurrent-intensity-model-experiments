from transformers import AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM
import torch, pandas as pd, numpy as np, warnings, functools
from rime.util import matrix_reindex, _to_cuda, empty_cache_on_exit, explode_user_titles
from tqdm import tqdm


class BayesLM:
    """ p(y|x) propto lm(x|y) * p(y), where x is the last item in user history
    and y is a candidate item.

    https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto
    """

    def __init__(self, item_df, max_num_candidates=None, batch_size=100,
                 prompt="a user will watch {y} after watching {x}",
                 item_pop_power=1, item_pop_pseudo=0.01, temperature=1, gamma=0,
                 candidate_selection_method=None, model_name='gpt2',  # bert-base-uncased
                 text_column_name='TITLE'):

        assert text_column_name in item_df, f"require {text_column_name} as data(y)"

        self.item_df = item_df.copy()
        self.item_df['log_p_y'] = item_pop_power * np.log(item_df['_hist_len'] + item_pop_pseudo)

        if max_num_candidates is None:
            warnings.warn("please set max_num_candidates, default=2 only for testing purposes")
            max_num_candidates = 2

        self.max_num_candidates = max_num_candidates
        self.batch_size = batch_size
        self.prompt = prompt
        self.temperature = temperature
        self.gamma = gamma
        self.text_column_name = text_column_name
 
        if candidate_selection_method is None:
            candidate_selection_method = 'greedy' if item_pop_power > 0 else 'sample'
        self.candidate_selection_method = candidate_selection_method

        # huggingface model initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert self.tokenizer.padding_side == 'right', "expect right padding"
        if model_name == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model_name.startswith('bert'):
            self.model = BertForMaskedLM.from_pretrained(model_name)
        else:  # gpt2
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # eval mode

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    @torch.no_grad()
    def _compute_log_p_x_given_y(self, Y, x, device):
        """ evaluate p_x_given_y for each y in Y; label = x.expand(...) """

        batch_size = len(Y)
        sequences = [self.prompt.format(y=y, x=x) for y in Y]
        inputs = self.tokenizer(sequences, padding=True, return_tensors='pt').to(device)
        targets = self.tokenizer(x, return_tensors='pt')['input_ids'].to(device)
        targets = torch.vstack([targets for _ in range(batch_size)])

        seq_len = inputs['attention_mask'].sum(1).tolist()
        target_len = targets.shape[1]

        if hasattr(self.model, "transformer"):  # gpt causal lm
            hidden_states = self.model.transformer(**inputs)[0]
            hidden_states = torch.vstack([x[n - target_len - 1: n - 1]
                                          for x, n in zip(hidden_states, seq_len)])
            logits = self.model.lm_head(hidden_states)

        elif hasattr(self.model, "bert"):  # bert [CLS] sequence [SEP], performs similarly
            targets = targets[:, 1:-1]
            target_len = target_len - 2

            hidden_states = self.model.bert(**inputs)[0]
            hidden_states = torch.vstack([x[n - target_len - 1: n - 1]  # [3-1-1 : 3-1]
                                          for x, n in zip(hidden_states, seq_len)])
            logits = self.model.cls(hidden_states)

        else:  # decoding non-target items can lead to 20% longer compute time
            logits = self.model(**inputs).logits
            logits = torch.vstack([x[n - target_len - 1: n - 1]
                                   for x, n in zip(logits, seq_len)])

        loss = self.loss(logits, targets.reshape(-1))
        return (-loss).reshape(targets.shape).mean(1).tolist()

    @functools.lru_cache(2)
    @empty_cache_on_exit
    def transform(self, D):
        """ generate score matrix by evaluating top or random items in test """

        explode_titles, splits, weights = explode_user_titles(
            D.user_in_test['_hist_items'], self.item_df[self.text_column_name], self.gamma)

        sorted_items = self.item_df[self.item_df.index.isin(D.item_in_test.index)] \
                           .sort_values('log_p_y', ascending=False, kind='mergesort')
        p_y = torch.as_tensor(sorted_items['log_p_y'].values).softmax(0).numpy()
        num_candidates = int(min(self.max_num_candidates, len(sorted_items)))

        with _to_cuda(self.model) as model:
            scores = []

            for x in tqdm(explode_titles.values):
                if self.candidate_selection_method == 'greedy':
                    ind = np.arange(num_candidates)
                else:
                    ind = np.random.choice(len(sorted_items), num_candidates, False, p_y)

                candidate_titles = sorted_items[self.text_column_name].values[ind]
                log_p_y_ind = sorted_items['log_p_y'].values[ind]

                log_p_x_given_y_ind = np.hstack([
                    self._compute_log_p_x_given_y(Y, x, model.device) for Y in
                    np.split(candidate_titles, range(0, num_candidates, self.batch_size)[1:])
                ])
                log_p_y_given_x_ind = log_p_x_given_y_ind / self.temperature + log_p_y_ind

                log_p_y_given_x = matrix_reindex(
                    log_p_y_given_x_ind, sorted_items.index[ind],
                    D.item_in_test.index, axis=0, fill_value=-np.inf)

                p_y_given_x = torch.as_tensor(log_p_y_given_x).softmax(0).numpy()
                scores.append(p_y_given_x)

        user_item_scores = np.vstack([w @ x for w, x in zip(
            np.split(weights, splits), np.split(np.vstack(scores), splits)
        )])

        return user_item_scores  # dense matrix with shape = user_in_test x item_in_test
