from pathlib import Path


def parse_misclassified(path):
    text_to_record = {}
    current = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        if line == '--------------------':
            if 'text' in current:
                text_to_record[current['text']] = current.copy()
            current = {}
            continue
        if ': ' in line:
            key, value = line.split(': ', 1)
            current[key] = value
    if 'text' in current:
        text_to_record[current['text']] = current.copy()
    return text_to_record


def print_records(title, records, task_a, task_b, limit=10):
    print('=' * 60)
    print(f'{title} sample count: {len(records)}')
    print('=' * 60)
    for i, item in enumerate(records[:limit], 1):
        text = item.get('text', '')
        task_a_record = task_a.get(text, {})
        task_b_record = task_b.get(text, {})
        true_class = task_a_record.get('true_class') or task_b_record.get('true_class', '')
        task_a_pred = task_a_record.get('pred_class', 'Correct')
        task_b_pred = task_b_record.get('pred_class', 'Correct')
        task_a_processed = task_a_record.get('processed_text', text)
        task_b_processed = task_b_record.get('processed_text', text)

        print(f'[{title} #{i}]')
        print('Original test text:', text)
        print('Task A processed text:', task_a_processed)
        print('Task B preprocessed text:', task_b_processed)
        print('Task A prediction:', task_a_pred)
        print('Task B prediction:', task_b_pred)
        print('Ground truth:', true_class)
        print('-' * 60)


def main():
    task_a_path = Path('THUCNews/data/taskA.misclassified')
    task_b_path = Path('THUCNews/data/taskB.misclassified')

    if not task_a_path.exists():
        raise FileNotFoundError(f'File not found: {task_a_path}')
    if not task_b_path.exists():
        raise FileNotFoundError(f'File not found: {task_b_path}')

    task_a = parse_misclassified(task_a_path)
    task_b = parse_misclassified(task_b_path)

    a_only = [task_a[text] for text in task_a if text not in task_b]
    b_only = [task_b[text] for text in task_b if text not in task_a]

    print_records('Misclassified in Task A but correct in Task B', a_only, task_a, task_b)
    print()
    print_records('Misclassified in Task B but correct in Task A', b_only, task_a, task_b)


if __name__ == '__main__':
    main()
