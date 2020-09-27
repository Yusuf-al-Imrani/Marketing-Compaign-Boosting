# Marketing-Compaign-Boosting
This is a data science project to promote a retailer marketing campaigns to reach more sales 

XYZ Group wants to leverage advanced analytics for boosting marketing campaign. You need to meet with the
customer to present your advanced analytics capabilities and position its value.

Overview
XYZ Group is one of the leading retailers’ industry in the region, with more than 50 branches across
the region. It runs multiple lines of business applications, mainly in the sport goods industry. They are
in the middle of their digital transformation journey and they want to keep leading the market by
satisfying their customers and meeting their expectation.

Business Environment and Goals
XYZ Group will releasing an advanced analytics RFP. Each of the vendors will be given a 30 min slot to
present their analytics capabilities and the value that their technology presents in front of us; including
the value from both a technical and business perspective.
The first part of the meeting will include people from two teams, the marketing team and the
management team. We want to see the value from using your platform for advanced analytics and the
data platform challenges that accompany it. Given the data sample we provided from our sales
databases, we want to know if using advanced analytics, you can help the marketing team. While the
marketing team will be interest to see if they can increase their efficiency, the management team want
to know if using advanced analytics may increase sales in general.
Finally, for another 30 min, our data scientist team is interested in the data science process you are
using and the details of the algorithm you are using to solve the proposed problem.
Tasks
We need a data scientist to discuss the following:

• Think about how you will be conducting the business conversation (purpose, agenda … etc.)

• Discuss and confirm your understanding of the organization’s business climate and goals

• Explore the problem and its impact to the company

• Communicate the business value of your advanced analytics platform offering that addresses
the problem or need the customer has communicated

• Address any customer objections

• Explaining the details of the machine learning model is crucial. Pick a model at your choice (e.g.
Logistic Regression) and be ready to go through the details of the algorithm.

• Close the meeting with clear next step(s)

• You will be evaluated based on communication and presentation skills


Data
The Sales Data Sample
The provided data represents information from a marketing campaign. We provided you information
about the product in the campaign/offer that was send to a specific customer and the convergence
result of this campaign (label attribute). We provided you we every attribute we have about the
product, use whatever you see make sense to solve the defined project.
Here is a list of the attributes:

• country: Country name


• article: 6 digit article number, as unique identifier of an article

• sales: total number of units sold in respective retail week

• regular_price: recommended retail price of the article

• current_price: current selling price (weighted average over the week)

• ratio: price ratio as current_price/regular_price, such that price discount is 1-
ratio

• retailweek: start date of the retailweek

• promo1: indicator for media advertisement, taking 1 in weeks of activation and 0
otherwise

• promo2: indicator for store events, taking 1 in weeks with events and 0 otherwise

• customer_id: customer unique identifier, one id per customer

• article: 6 digit article number, as unique identifier of an article

• productgroup: product group the article belongs to

• category: product category the article belongs to

• cost: total costs of the article (assumed to be fixed over time)

• style: description of article design

• sizes: size range in which article is available

• gender: gender of target consumer of the article

• rgb_*_main_color: intensity of the red (r), green (g), and blue (b) primaries of the article‘s
main color, taking values [0,250]

• rgb_*_sec_color: intensity of the red (r), green (g), and blue (b) primaries of the article‘s
secondary color, taking values [0,250]

• label: advertisement result after offering/sending/presenting the offer to the
customer. 0 means the customer did not buy and 1 means the costomer did buy.
