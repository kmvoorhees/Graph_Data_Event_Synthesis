import json
import numpy as np
import pandas as pd

SURVEY_RATE = 1
RESPONSE_RATE = .25

with open('hcahps.json') as hcahps_json:
    hcahps_questions = json.load(hcahps_json) 
ip_visits = pd.read_csv('2023-05-18InpatientTest.csv')
ip_visits.columns = ip_visits.columns.str.lower()

effects = {
    'bad_provider' : [
        {
            'questions': 'all',
            'effect_size': -.5
        }
    ],
    'good_provider' : [
        {
            'questions': 'all',
            'effect_size': .3
        }
    ],
    'bad_outcome' : [
        {
            'questions': 'all',
            'effect_size': -.5
        }
    ],
    'good_outcome' : [
        {
            'questions': 'all',
            'effect_size': .1
        }
    ],
}

survey = hcahps_questions

for question in survey.keys():
    survey[question]['effects'] = {}

for effect, subeffects in effects.items():
    for subeffect in subeffects:
        if subeffect['questions'] == 'all':
            questions = survey.keys()
        else:
            questions = subeffect['questions']
        for question in questions:
            survey[question]['effects'][effect] = subeffect['effect_size']


sample_size = round(len(ip_visits) * SURVEY_RATE * RESPONSE_RATE)
rng = np.random.default_rng(1234)

event_sample = ip_visits.sample(sample_size, random_state = rng)

def mutate_distribution(distribution, effect_size):
    shift_factor = 2 / len(distribution)
    final_effect = effect_size * np.dot(distribution, range(1, len(distribution) + 1))
    shift_high = final_effect * shift_factor
    shift_low = - final_effect_size * shift_factor / (len(distribution) - 1)
    low_dist = [val + shift_low for val in distribution[:-1]]
    high_val = distribution[-1] + shift_high
    final_dist = low_dist.append(high_value)
    if min(final_dist) < 0 or max(final_dist) > 1:
        raise ValueError('Effect size is too large for distribution')
    return final_dist


experience_event_rows = []

for index, event in event_sample.iterrows():
    for question_name, question_spec in survey.items():
        total_effect_size = 0
        for effect_name, effect_size in question_spec['effects'].items():
            total_effect_size = total_effect_size + effect_size * event[effect_name]
        final_prob = mutate_distribution(question['base_prob'], total_effect_size)
        assigned_value = rng.choice(question['responses'], p = final_prob)
        new_row = {
            'member_id': event.member_id,
            'event_id': event.event_id,
            'instrument': 'HCAHPS',
            'survey_item': question_name,
            'response': assigned_value
        }
        experience_event_rows.append(new_row)

experience_events = pd.DataFrame(experience_event_rows)
experience_events.to_csv('hcahps_generated_scores.csv')
