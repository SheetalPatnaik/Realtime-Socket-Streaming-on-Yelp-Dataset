import json
import socket
import time
import pandas as pd

def handle_date(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

def send_data_over_socket(file_path, host='0.0.0.0', port=9999, chunk_size=2):
    last_sent_index = 0
    try:
        with open("checkpoint.txt", "r") as checkpoint:
            last_sent_index = int(checkpoint.read().strip())
    except FileNotFoundError:
        pass

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print(f"Listening for connections on {host}:{port}")

    while True:
        conn, addr = s.accept()
        print(f"Connection from {addr}")
        try:
            with open(file_path, 'r') as file:
                # Skip lines already sent
                for _ in range(last_sent_index):
                    next(file)

                records = []
                for line in file:
                    records.append(json.loads(line))
                    if len(records) == chunk_size:
                        chunk = pd.DataFrame(records)
                        print(chunk)
                        for record in chunk.to_dict(orient='records'):
                            serialize_data = json.dumps(record, default=handle_date).encode('utf-8')
                            conn.sendall(serialize_data + b'\n')
                            time.sleep(5)  # Delay between messages
                            last_sent_index += 1
                            with open("checkpoint.txt", "w") as checkpoint:
                                checkpoint.write(str(last_sent_index))

                        records = []
        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected.")
        finally:
            conn.close()
            print("Connection closed")

if __name__ == "__main__":
    send_data_over_socket("datasets/yelp_academic_dataset_review.json")
