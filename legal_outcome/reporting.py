"""Plain console tables and persistent machine-readable result files."""
import csv
import io
import json
import os
from pathlib import Path


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_file(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_file(path, headers, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    width = [max(len(str(header)), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    line = lambda row: ' | '.join(str(value).ljust(size) for value, size in zip(row, width))
    return '\n'.join([line(headers), '-+-'.join('-' * size for size in width), *(line(row) for row in rows)])


def percentage(value):
    return 'n/a' if value is None else f'{100 * value:.2f}%'


def history_table(history):
    rows = []
    for row in history:
        val, train = row.get('validation', {}), row.get('training', {})
        rows.append([row['epoch'], f"{row['training_loss']:.4f}",
                     f"{val['loss']:.4f}" if 'loss' in val else 'n/a',
                     percentage(train.get('micro_f1')), percentage(val.get('micro_f1')),
                     percentage(val.get('macro_f1')), 'YES' if row.get('selected') else ''])
    return table(['Epoch', 'Train loss', 'Val loss', 'Train micro F1', 'Val micro F1', 'Val macro F1', 'New best'], rows)


def write_history(directory, history):
    directory = Path(directory)
    json_file(directory / 'history.json', history)
    headers = ['epoch', 'training_loss', 'training_evaluation_loss', 'validation_loss', 'training_micro_f1',
               'validation_micro_f1', 'validation_macro_f1', 'validation_exact_match_accuracy', 'new_best']
    rows = [[r['epoch'], r['training_loss'], r.get('training', {}).get('loss'), r.get('validation', {}).get('loss'),
             r.get('training', {}).get('micro_f1'), r.get('validation', {}).get('micro_f1'),
             r.get('validation', {}).get('macro_f1'), r.get('validation', {}).get('exact_match_accuracy'), r.get('selected')]
            for r in history]
    csv_file(directory / 'history.csv', headers, rows)
    atomic_text(directory / 'training-summary.txt', history_table(history) + '\n\nScores use percentages here and fractions in JSON/CSV. Training minibatch loss uses training-mode dropout.\n')


def result_table(result):
    names = [('micro_precision', 'Micro precision'), ('micro_recall', 'Micro recall'), ('micro_f1', 'Micro F1'),
             ('macro_precision', 'Macro precision'), ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro F1'),
             ('accuracy', 'Accuracy'), ('exact_match_accuracy', 'Exact-match accuracy'), ('hamming_loss', 'Hamming loss'),
             ('micro_jaccard', 'Micro Jaccard'), ('micro_pr_auc', 'Micro PR-AUC (trapezoid)'),
             ('micro_average_precision', 'Micro average precision'), ('micro_roc_auc', 'Micro ROC-AUC')]
    rows = [[label, percentage(result[key])] for key, label in names if key in result]
    if 'loss' in result:
        rows.append(['Evaluation loss', f"{result['loss']:.4f}"])
    return table(['Measure', 'Result'], rows)


def save_result_files(directory, result):
    directory = Path(directory)
    json_file(directory / 'test-results.json', result)
    per_label = result.get('per_label', {})
    label_rows = [[label, values['support'], values['precision'], values['recall'], values['f1']]
                  for label, values in per_label.items()]
    csv_file(directory / 'per-article-results.csv', ['article', 'support', 'precision', 'recall', 'f1'], label_rows)
    scalar_rows = [[key, value] for key, value in result.items() if value is None or isinstance(value, (str, int, float, bool))]
    csv_file(directory / 'test-summary.csv', ['measure', 'value'], scalar_rows)
    article_table = table(['Article', 'Support', 'Precision', 'Recall', 'F1'],
                          [[label, v['support'], percentage(v['precision']), percentage(v['recall']), percentage(v['f1'])]
                           for label, v in per_label.items()])
    text = (f"TEST RESULTS | cases: {result['examples']} | selected epoch: {result.get('selected_epoch', 'unknown')}\n\n"
            + result_table(result) + '\n\n' + article_table
            + '\n\nMicro F1 is not exact-match accuracy. Hamming loss is better when lower.\n'
            + 'PR-AUC uses trapezoidal integration; average precision is a separate measure.\n')
    if 'input_limit_note' in result:
        text += result['input_limit_note'] + '\n'
    atomic_text(directory / 'test-report.txt', text)
    return text


class Tee:
    """Keep live output and append it to a local UTF-8 log."""
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        count = self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return count
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return self.stream.isatty()
    def __getattr__(self, key):
        return getattr(self.stream, key)
