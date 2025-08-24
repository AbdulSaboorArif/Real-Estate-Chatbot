# ############################## Real Estate AI Chatbot Agent ##############################

import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, RunConfig
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from dynamic.dynamic_Context import ai_chatbot_agent_instructions, sale_agent_instructions, rent_agent_instructions, website_agent_instructions, contact_agent_instructions
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv(find_dotenv())

# FastAPI app
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Next.js frontend URL
    "https://real-estate-website-snowy-eight.vercel.app/",  # Vercel deployment URL
    "https://tester-chatbot.onrender.com/chat",  # Render deployment URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Gemini API setup
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model configuration
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",  # Corrected model name
    openai_client=external_provider,
)

run_config = RunConfig(
    model=model,
    model_provider=external_provider,
    tracing_disabled=True,
)


# Function Tools
@function_tool
def sale_properties(query: str) -> str:
    """Tool to get information about properties for sale."""
    print(f"Sale properties query: {query}")
    return  [
    {
      "name": "Modern Luxury Villa",
      "description": "An architectural masterpiece featuring floor-to-ceiling glass, open concept living, and an infinity-edge pool overlooking Beverly Hills.",
      "price": 2500000,
      "type": "sale",
      "typeofproperty": "villa",
      "location": "Beverly Hills, CA",
      "beds": 5,
      "baths": 4,
      "sqft": 4200,
      "features": ["Open kitchen", "Smart home", "Home theater", "3-car garage"]
    },
    {
      "name": "Luxury Family Home",
      "description": "Spacious family residence with bright interiors, landscaped backyard, and a quiet cul-de-sac address.",
      "price": 750000,
      "type": "sale",
      "typeofproperty": "single_family",
      "location": "Suburban Heights, CA",
      "beds": 4,
      "baths": 3,
      "sqft": 2800,
      "features": ["Family room", "Breakfast nook", "Primary suite balcony"]
    },
    {
      "name": "Investment Property",
      "description": "Turn-key investment near the university, strong rental history and low vacancy.",
      "price": 450000,
      "type": "sale",
      "typeofproperty": "townhouse",
      "location": "University Area, CA",
      "beds": 3,
      "baths": 2,
      "sqft": 1600,
      "features": ["Updated kitchen", "Hardwood floors"]
    },
    {
      "name": "Historic Townhouse",
      "description": "Classic 1925 townhouse meticulously updated while preserving original character.",
      "price": 1200000,
      "type": "sale",
      "typeofproperty": "townhouse",
      "location": "Historic District, CA",
      "beds": 5,
      "baths": 4,
      "sqft": 3200,
      "features": ["Crown molding", "Chef kitchen", "Library"]
    },
    {
      "name": "Modern Condo",
      "description": "Contemporary condo in the heart of downtown with amenities and parking.",
      "price": 380000,
      "type": "sale",
      "typeofproperty": "condo",
      "location": "Downtown, CA",
      "beds": 2,
      "baths": 2,
      "sqft": 1100,
      "features": ["Balcony", "Walk-in closet"]
    },
    {
      "name": "Waterfront Estate",
      "description": "Stunning waterfront estate with private dock and guest house.",
      "price": 2500000,
      "type": "sale",
      "typeofproperty": "estate",
      "location": "Harbor View, CA",
      "beds": 6,
      "baths": 5,
      "sqft": 4500,
      "features": ["Guest house", "Private dock", "Gourmet kitchen"]
    },
    {
      "name": "Mountain View Cabin",
      "description": "Cozy cabin retreat with panoramic mountain views and modern finishes.",
      "price": 590000,
      "type": "sale",
      "typeofproperty": "single_family",
      "location": "Highland Ridge, CO",
      "beds": 3,
      "baths": 2,
      "sqft": 1700,
      "features": ["Fireplace", "Wraparound deck"]
    }
  ],

@function_tool
def rent_properties(query: str) -> str:
    """Tool to get information about properties for rent."""
    print(f"Rent properties query: {query}")
    return [
    {
      "name": "Waterfront Condo",
      "description": "Stylish condo with a private balcony facing the harbor, concierge service, and residents-only fitness center.",
      "price": 3200,
      "type": "rent",
      "typeofproperty": "condo",
      "location": "Harbor View, CA",
      "beds": 2,
      "baths": 2,
      "sqft": 1400,
      "features": ["Balcony", "Chef kitchen", "In-unit laundry"]
    },
    {
      "name": "Cozy Studio Loft",
      "description": "Charming loft with exposed brick and high ceilings in the vibrant Arts District.",
      "price": 1800,
      "type": "rent",
      "typeofproperty": "studio",
      "location": "Arts District, CA",
      "beds": 1,
      "baths": 1,
      "sqft": 800,
      "features": ["High ceilings", "Exposed brick"]
    },
    {
      "name": "Modern Downtown Apartment",
      "description": "Bright corner unit apartment with city skyline views and quick access to transit.",
      "price": 2500,
      "type": "rent",
      "typeofproperty": "apartment",
      "location": "Downtown, City Center",
      "beds": 2,
      "baths": 2,
      "sqft": 1200,
      "features": ["Floor-to-ceiling windows", "Quartz counters"]
    },
    {
      "name": "Family Townhouse",
      "description": "Three-bedroom townhouse with private patio and community park access.",
      "price": 4500,
      "type": "rent",
      "typeofproperty": "townhouse",
      "location": "Suburban Heights, CA",
      "beds": 3,
      "baths": 3,
      "sqft": 1800,
      "features": ["Private patio", "Attached garage"]
    },
    {
      "name": "Luxury Penthouse",
      "description": "Opulent penthouse with private elevator access and skyline terrace.",
      "price": 8000,
      "type": "rent",
      "typeofproperty": "apartment",
      "location": "Financial District, CA",
      "beds": 3,
      "baths": 3,
      "sqft": 2200,
      "features": ["Private elevator", "Terrace", "Wine fridge"]
    },
    {
      "name": "Garden Apartment",
      "description": "Serene garden-level apartment with direct courtyard access.",
      "price": 2200,
      "type": "rent",
      "typeofproperty": "apartment",
      "location": "Park View, CA",
      "beds": 2,
      "baths": 1,
      "sqft": 1100,
      "features": ["Courtyard access", "Breakfast bar"]
    },
    {
      "name": "Commercial Office Suite",
      "description": "Flexible Class-A office suite with 12 private offices, reception, and kitchen.",
      "price": 12000,
      "type": "rent",
      "typeofproperty": "commercial",
      "location": "Midtown, CA",
      "beds": 0,
      "baths": 2,
      "sqft": 5000,
      "features": ["Conference room", "Server room", "Reception"]
    }
  ],


@function_tool
def web_about(query: str) -> str:
    """Tool to get information about the real estate website basesd on user qurerie."""
    print(f"Website info query: {query}")
    return """
    The Real Estate Website modern platform designed to cater to the luxury real estate market. Its primary purpose is to provide an elegant, user-centric, and responsive interface that facilitates the discovery, exploration, and engagement with high-end real estate properties. The website aims to bridge the gap between potential buyers, renters, or investors and a real estate company by offering a seamless digital experience. With a dark-themed aesthetic inspired by luxury, the website combines visual appeal with functionality, enabling users to browse properties, learn about the company, connect through a contact form, and stay updated via newsletters and social media. It serves as a digital storefront for a real estate business, emphasizing professionalism, accessibility, and user engagement across various devices.
Services Offered
The website provides a comprehensive set of services tailored to meet the needs of users interested in real estate, including:

Property Showcase: The core service of the website is its ability to display featured properties in an organized and visually appealing manner. Each property listing includes detailed information such as high-quality images, pricing, location details, and other relevant specifications, allowing users to explore available real estate options effectively.
Property Search with Filters: The website offers a robust search functionality that enables users to find properties based on specific criteria, such as location, price range, or property type. Filters enhance the user experience by allowing refined searches to match individual preferences.
Contact Form: A fully functional contact form with validation is provided, enabling users to reach out to the real estate company directly. This form is designed to capture inquiries, feedback, or requests for property viewings, ensuring seamless communication.
Company Information and Team Profiles: The "About" section provides detailed insights into the real estate company, its mission, and its team members. This service helps build trust by showcasing the expertise and professionalism of the company’s staff.
Customer Testimonials: The website includes a dedicated section for customer reviews and ratings, offering social proof to potential clients. This service highlights positive experiences from previous customers, enhancing credibility.
Newsletter Subscription: Users can subscribe to a newsletter via an email subscription form, allowing them to receive updates on new property listings, market trends, or company news.
Social Media Integration: The website incorporates social media links and sharing options, enabling users to follow the company on various platforms or share property listings with their networks, increasing engagement and reach.

Features of the Website
The website is equipped with 10 distinct features that enhance its functionality, user experience, and aesthetic appeal:

Modern Design: The website adopts a dark theme with luxury aesthetics, creating a visually striking and sophisticated interface that aligns with the high-end real estate market. The design emphasizes clean lines, elegant typography, and a premium look.
Fully Responsive: Optimized for all device sizes, the website ensures a seamless experience across mobile devices (screen width < 640px), tablets (640px - 1024px), and desktops (> 1024px). This responsiveness guarantees accessibility for users regardless of the device they use.
Interactive Components: The website incorporates dynamic elements such as hover effects, animations, and smooth transitions to enhance user engagement and provide a modern, interactive browsing experience.
Property Showcase: A dedicated section displays featured properties in a grid layout, complete with high-quality images, hover effects, and detailed information such as price, location, and property specifications. This feature is central to the website’s purpose of promoting real estate listings.
Contact Form: The contact form is multi-field, user-friendly, and includes validation to ensure accurate submissions. It serves as a direct communication channel for users to inquire about properties or services.
Team Section: The website includes a section showcasing company team member profiles, complete with photos and brief descriptions, fostering trust and transparency with potential clients.
Testimonials: A testimonials section highlights customer reviews and ratings, providing authentic feedback to build confidence in the company’s services.
Search Functionality: The property search feature allows users to filter listings based on specific parameters, making it easier to find properties that meet their needs.
Newsletter Subscription: An email subscription form enables users to sign up for newsletters, keeping them informed about new listings and company updates.
Social Media Integration: The website integrates social media links and sharing capabilities, allowing users to connect with the company on platforms like Twitter, Instagram, or Facebook and share properties with others.

Additional Technical Details
The website is built using a modern tech stack to ensure performance, scalability, and maintainability:

Framework: Next.js 14 with App Router for server-side rendering and optimized performance.
Language: TypeScript for type-safe and maintainable code.
Styling: Tailwind CSS for rapid, customizable, and responsive styling.
Icons: Lucide React for modern, scalable icons.
Images: Next.js Image Optimization for fast-loading, high-quality visuals.
Fonts: Inter from Google Fonts for clean and professional typography.

The project structure is organized to facilitate development and customization, with key directories such as app/ for core components, components/ for reusable UI elements, and public/ for static assets. The website can be customized by updating the color scheme in tailwind.config.js, replacing images in components, or modifying content such as company details and property information.
Conclusion
The Real Estate Website is a feature-rich, responsive, and visually appealing platform designed to showcase luxury properties and facilitate user engagement with a real estate company. Its services, including property showcases, search functionality, and direct communication channels, cater to users seeking premium real estate solutions. With 10 carefully crafted features, the website delivers a seamless and professional experience, making it an effective tool for both the company and its clients."""


@function_tool
def contact_info(query: str) -> str:
    """Tool to get contact information and schedule meetings."""
    print(f"Contact info query: {query}")
    return """
    For assistance, contact us at:
    - Email: support@yourrealestatewebsite.com
    - Phone: +1-123-456-7890
    Please mention the property name or type you're interested in, and we'll arrange a consultation at your preferred time.
    """


# Sale Agent
sale_agent = Agent(
    name="SaleAgent",
    instructions=sale_agent_instructions,
    model=model,
    tools=[sale_properties],
)

# Rent Agent
rent_agent = Agent(
    name="RentAgent",
    instructions=rent_agent_instructions,
    model=model,
    tools=[rent_properties],
)

# Website Agent
web_agent = Agent(
    name="WebsiteAgent",
    instructions=website_agent_instructions,
    model=model,
    tools=[web_about],
)

# Contact Agent
contact_agent = Agent(
    name="ContactAgent",
    instructions= contact_agent_instructions,
    model=model,
    tools=[contact_info],
)

# Real Estate AI Chatbot Orchestrator Agent
ai_chatbot_agent = Agent(
    name="RealEstateAIChatbotAgent",
    instructions=ai_chatbot_agent_instructions,
    model=model,
    tools=[
        sale_agent.as_tool(
            tool_name="SaleAgent",
            tool_description="Tool to provide information about properties for sale based on user queries. Your task is to get the information of sale properties accordingly to the user need.",
        ),
        rent_agent.as_tool(
            tool_name="RentAgent",
            tool_description="Tool to provide information about properties for rent based on user queries. Your task is to get the information of rent properties accordingly to the user need.",
        ),
        web_agent.as_tool(
            tool_name="WebsiteAgent",
            tool_description="Tool to provide information about the website based on user queries. Your task is to get the information of website accordingly to the user need.",
        ),
        contact_agent.as_tool(
            tool_name="ContactAgent",
            tool_description="Tool to handle contact information and meeting scheduling for users interested in properties.",
        ),
    ],
)


class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    FastAPI endpoint to handle chat messages.
    """
    try:
        result = await Runner.run(ai_chatbot_agent, chat_message.message, run_config=run_config)
        return {"response": result.final_output}
    except ValueError as ve:
        return {"error": f"Invalid input: {str(ve)}"}
    except Exception as e:
        return {"error": "An unexpected error occurred. Please try again later."}