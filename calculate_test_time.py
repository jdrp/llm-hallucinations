import os


def main() -> None:
    
    total = 0
    logs_dir = 'logs'

    for filename in os.listdir(logs_dir):
        if filename.endswith('.log'):
            try:
                with open(os.path.join(logs_dir, filename), 'rb') as f:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                    last_line = f.readline().decode()

                    if last_line.startswith('Finished after'):
                        runtime = float(last_line.split('Finished after ')[1].split(' seconds')[0])
                        total += runtime
            except (UnicodeDecodeError, ValueError, IndexError) as e:
                print(f"Skipping file {filename} due to error: {e}")
            except OSError:
                print(f"Error reading file {filename}")

    total_minutes = total / 60
    total_hours = int(total_minutes // 60)
    remaining_minutes = int(total_minutes % 60)

    print(f"Total runtime: {total_hours} hours and {remaining_minutes} minutes")


if __name__ == '__main__':
    main()
