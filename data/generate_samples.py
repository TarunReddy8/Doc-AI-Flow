"""
Generate sample document images for testing the DocAI pipeline.
Run: python data/generate_samples.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap


def create_invoice_image(output_path: str):
    """Generate a realistic-looking invoice image."""
    img = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font_title = ImageFont.load_default()
        font_header = font_title
        font_body = font_title
        font_small = font_title

    # Header
    draw.rectangle([0, 0, 800, 80], fill="#1a365d")
    draw.text((40, 22), "ACME CLOUD SERVICES", fill="white", font=font_title)
    draw.text((580, 35), "INVOICE", fill="white", font=font_header)

    # Invoice details
    y = 110
    draw.text((40, y), "Invoice Number:", fill="#666", font=font_small)
    draw.text((180, y), "INV-2024-0847", fill="black", font=font_body)
    draw.text((450, y), "Date:", fill="#666", font=font_small)
    draw.text((500, y), "March 15, 2024", fill="black", font=font_body)
    y += 25
    draw.text((450, y), "Due:", fill="#666", font=font_small)
    draw.text((500, y), "April 15, 2024", fill="black", font=font_body)

    # From / To
    y = 190
    draw.text((40, y), "FROM:", fill="#1a365d", font=font_header)
    draw.text((400, y), "BILL TO:", fill="#1a365d", font=font_header)
    y += 25
    draw.text((40, y), "Acme Cloud Services", fill="black", font=font_body)
    draw.text((400, y), "Widget Corp", fill="black", font=font_body)
    y += 20
    draw.text((40, y), "123 Tech Boulevard", fill="#666", font=font_small)
    draw.text((400, y), "456 Innovation Drive", fill="#666", font=font_small)
    y += 18
    draw.text((40, y), "San Francisco, CA 94105", fill="#666", font=font_small)
    draw.text((400, y), "Austin, TX 78701", fill="#666", font=font_small)

    # Table header
    y = 320
    draw.rectangle([40, y, 760, y + 35], fill="#f0f4f8")
    draw.text((50, y + 8), "Description", fill="#1a365d", font=font_header)
    draw.text((420, y + 8), "Qty", fill="#1a365d", font=font_header)
    draw.text((500, y + 8), "Unit Price", fill="#1a365d", font=font_header)
    draw.text((650, y + 8), "Total", fill="#1a365d", font=font_header)

    # Table rows
    items = [
        ("Cloud Hosting (monthly)", "1", "$2,400.00", "$2,400.00"),
        ("API Calls (10K bundle)", "3", "$150.00", "$450.00"),
        ("Premium Support Plan", "1", "$500.00", "$500.00"),
        ("SSL Certificate (annual)", "2", "$75.00", "$150.00"),
    ]

    y += 40
    for desc, qty, price, total in items:
        draw.text((50, y + 5), desc, fill="black", font=font_body)
        draw.text((430, y + 5), qty, fill="black", font=font_body)
        draw.text((510, y + 5), price, fill="black", font=font_body)
        draw.text((650, y + 5), total, fill="black", font=font_body)
        y += 35
        draw.line([40, y, 760, y], fill="#e2e8f0", width=1)

    # Totals
    y += 30
    draw.line([500, y, 760, y], fill="#e2e8f0", width=1)
    y += 10
    draw.text((500, y), "Subtotal:", fill="#666", font=font_body)
    draw.text((650, y), "$3,500.00", fill="black", font=font_body)
    y += 28
    draw.text((500, y), "Tax (8.5%):", fill="#666", font=font_body)
    draw.text((650, y), "$297.50", fill="black", font=font_body)
    y += 28
    draw.rectangle([490, y - 5, 760, y + 25], fill="#1a365d")
    draw.text((500, y), "TOTAL:", fill="white", font=font_header)
    draw.text((640, y), "$3,797.50", fill="white", font=font_header)

    # Payment terms
    y += 60
    draw.text((40, y), "Payment Terms: Net 30", fill="#666", font=font_small)
    y += 20
    draw.text((40, y), "Please make payment to: Acme Cloud Services", fill="#666", font=font_small)

    # Footer
    draw.rectangle([0, 1050, 800, 1100], fill="#f0f4f8")
    draw.text((40, 1065), "Thank you for your business!", fill="#666", font=font_small)
    draw.text((500, 1065), "acme-cloud.example.com", fill="#1a365d", font=font_small)

    img.save(output_path, quality=95)
    print(f"Created: {output_path}")


def create_contract_image(output_path: str):
    """Generate a realistic-looking contract image."""
    img = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font_title = ImageFont.load_default()
        font_header = font_title
        font_body = font_title

    y = 50
    draw.text((200, y), "MASTER SERVICE AGREEMENT", fill="#1a365d", font=font_title)

    y = 100
    contract_text = [
        ("PARTIES", "This Master Service Agreement ('Agreement') is entered into as of January 1, 2024, by and between:"),
        ("", "Party A: Acme Cloud Services, Inc., a Delaware corporation ('Provider')"),
        ("", "Party B: Widget Corp, a Texas corporation ('Client')"),
        ("TERM", "This Agreement shall commence on January 1, 2024 and shall continue until December 31, 2025, unless terminated earlier."),
        ("SERVICES", "Provider agrees to deliver cloud hosting, API infrastructure, and technical support services as described in Exhibit A."),
        ("COMPENSATION", "Client shall pay Provider a total contract value of $240,000.00 USD, payable in monthly installments of $10,000.00."),
        ("TERMINATION", "Either party may terminate this Agreement with 30 days written notice. In the event of a material breach, the non-breaching party may terminate immediately."),
        ("GOVERNING LAW", "This Agreement shall be governed by and construed in accordance with the laws of the State of California."),
        ("CONFIDENTIALITY", "Both parties agree to maintain the confidentiality of all proprietary information exchanged during the term of this Agreement."),
    ]

    for title, text in contract_text:
        if title:
            y += 15
            draw.text((60, y), title, fill="#1a365d", font=font_header)
            y += 22
        wrapped = textwrap.wrap(text, width=85)
        for line in wrapped:
            draw.text((60, y), line, fill="#333", font=font_body)
            y += 18
        y += 5

    # Signatures
    y += 30
    draw.line([60, y, 340, y], fill="#999", width=1)
    draw.line([440, y, 720, y], fill="#999", width=1)
    y += 5
    draw.text((60, y), "Provider Signature", fill="#666", font=font_body)
    draw.text((440, y), "Client Signature", fill="#666", font=font_body)
    y += 20
    draw.text((60, y), "Date: _______________", fill="#666", font=font_body)
    draw.text((440, y), "Date: _______________", fill="#666", font=font_body)

    img.save(output_path, quality=95)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "sample_docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    create_invoice_image(str(output_dir / "sample_invoice.png"))
    create_contract_image(str(output_dir / "sample_contract.png"))
    print("\nSample documents generated! Use them to test the pipeline.")
