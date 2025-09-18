from pipeline import store_plain_text

if __name__ == "__main__":
    text = """NovaTech Solutions, established in 2018, is based in New York City.
The company focuses on cybersecurity, blockchain development, and enterprise cloud infrastructure.
Its CEO, Michael Chen, is known for driving innovation, while CTO Sarah Patel leads the R&D division.
In 2024, NovaTech released its breakthrough security platform, ShieldX, which uses AI-driven threat detection.
    """
    store_plain_text(text, user_id="00000000-0000-0000-0000-000000000000")
