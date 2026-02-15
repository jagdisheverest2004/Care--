from safety_engine import DrugSafetyEngine

def test_manual():
    print("Loading your NEW Trained Model...")
    # Ensure this matches your folder name
    engine = DrugSafetyEngine(model_name="models/biobert_ddi") 
    
    print("\n--- TEST 1: The Specific Bug (Should be SAFE) ---")
    # This was giving 0.52 (Danger) before. Now it should be very low.
    result = engine.check_interaction("Metformin", "Verteporfin")
    print(f"Metformin + Verteporfin: {result}")

    print("\n--- TEST 2: A Known Danger (Should be UNSAFE) ---")
    # Aspirin + Warfarin is a classic interaction
    result = engine.check_interaction("Warfarin", "Aspirin")
    print(f"Warfarin + Aspirin: {result}")

if __name__ == "__main__":
    test_manual()