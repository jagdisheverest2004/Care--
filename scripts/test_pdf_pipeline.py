from data_loader import DocumentLoader
from summarizer import DocumentSummarizer  # <-- Import the new class
import os

def test_pipeline():
    # 1. Initialize our tools
    loader = DocumentLoader()
    doc_ai = DocumentSummarizer()

    # Create a dummy long text file for testing
    # A realistic, dense clinical document
    long_medical_text = """
    CHIEF COMPLAINT: Shortness of breath and swelling in the lower extremities.
    
    HISTORY OF PRESENT ILLNESS: The patient is a 68-year-old female with a known history of congestive heart failure (CHF), hypertension, and chronic kidney disease (Stage 3). She presents to the clinic today reporting a progressive worsening of dyspnea on exertion over the past four days. She notes that she can now only walk about 10 feet before needing to stop and catch her breath. She also reports a 5-pound weight gain over the last week and increasing 2+ pitting edema in both ankles extending up to her mid-calves. She denies any chest pain, palpitations, or recent illnesses. She admits she attended a family barbecue over the weekend and consumed foods high in sodium, including hot dogs and potato chips. She has been taking her prescribed Carvedilol 12.5 mg twice daily, but ran out of her Furosemide (Lasix) three days ago.
    
    OBJECTIVE/VITALS: 
    Blood Pressure: 165/95 mmHg. Heart Rate: 92 bpm, regular. Respiratory Rate: 22 breaths/min. Oxygen Saturation: 93% on room air. Temperature: 98.6 F. Weight: 182 lbs (up 5 lbs from baseline).
    Physical Exam reveals jugular venous distention (JVD) at 45 degrees. Chest auscultation reveals bibasilar crackles extending roughly one-third of the way up lung fields. Heart sounds are normal with no S3 or murmurs appreciated. Extremities show bilateral 2+ pitting edema to the mid-calf. 
    
    ASSESSMENT: 
    1. Acute exacerbation of chronic systolic heart failure, likely triggered by dietary non-compliance and medication non-adherence (missed diuretic doses). 
    2. Uncontrolled hypertension, likely secondary to fluid overload.
    
    PLAN: 
    1. Administer Furosemide 40 mg IV push in the clinic now. Observe patient for 2 hours to monitor urine output and respiratory status.
    2. Provide a strict warning and education regarding a low-sodium diet (< 2 grams daily). 
    3. Restart oral Furosemide 40 mg daily upon discharge. 
    4. Patient is instructed to weigh herself daily and call the clinic if she gains more than 2 lbs in a day or 5 lbs in a week.
    5. Follow up appointment scheduled in 3 days for a recheck of vitals, weight, and a basic metabolic panel (BMP) to check renal function and potassium levels.
    """

    with open("temp_test.txt", "w") as f:
        f.write(long_medical_text)

    print("\n--- Testing Map-Reduce Pipeline ---")
    
    # Simulate Chunking
    words = long_medical_text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
    print(f"Created {len(chunks)} chunks from document.")

    # Run Map-Reduce
    final_summary = doc_ai.summarize_long_document(chunks)

    print("\n" + "="*50)
    print("🏆 FINAL MASTER SUMMARY:")
    print(final_summary)
    print("="*50 + "\n")

    if os.path.exists("temp_test.txt"):
        os.remove("temp_test.txt")

if __name__ == "__main__":
    test_pipeline()