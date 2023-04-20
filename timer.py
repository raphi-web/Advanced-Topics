import datetime
class Timer():
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start_time = datetime.datetime.now()
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        end_time = datetime.datetime.now()
        diff = end_time - self.start_time
        print(self.label,diff.total_seconds()*1000,"[μs]")

if __name__ == "__main__":
    with Timer("Execution Time:"):
        [i for i in range(100000)]