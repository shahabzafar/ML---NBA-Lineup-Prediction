from src.evaluate_test_data import main as evaluate_main

if __name__ == "__main__":
    print("Running NBA test data evaluation...")
    metrics = evaluate_main()
    print("\nEvaluation complete! View results at http://localhost:5000/evaluation") 