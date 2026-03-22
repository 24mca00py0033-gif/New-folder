from pipeline import MisinformationPipeline
import traceback

def test():
    try:
        p = MisinformationPipeline()
        res = p.run_simulation()
        print("Pipeline finished.")
        try:
            print("Visualizing...")
            path = p.network.visualize_spread_analysis(res)
            print("Success! Path:", path)
        except Exception as e:
            print("Visualization failed:")
            traceback.print_exc()
    except Exception as e:
        print("Pipeline fail:")
        traceback.print_exc()

if __name__ == "__main__":
    test()