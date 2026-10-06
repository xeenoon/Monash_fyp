#include "dialogue.h"

static const DialogueEntry entries[] = {
	{"mumble_1", {"Orrum. Shabba nok. Frindle-voss."}, 1},
	{"mumble_2", {"Blem namaru. Skrong. Skrong."}, 1},
	{"mumble_3", {"Yabba throm. Nib-nib-nib. HOOOOOM."}, 1},
	{"warning",
	 {"Stay away from the snake lady. Especially if she tells you she's worked on herself."},
	 1},
	{"confession",
	 {"The snake lady? My wife. Former wife. Current emergency contact, owing to a clerical "
	  "dispute. Married forty-three years. Six if you subtract the time I spent petrified beside "
	  "the conservatory.",
	  "We met at a municipal hearing about damp. She spoke beautifully. Her hair disagreed on "
	  "several points. I proposed that evening. Seven snakes accepted. Two wanted to see my "
	  "accounts.",
	  "Our wedding was modest. Seventy guests, fourteen mirrors, one nervous photographer. Her "
	  "mother gave us a gravy boat. Her mother was also, temporarily, the gravy boat. I never "
	  "understood that side of the family.",
	  "Then I met a spider named Deborah behind the airing cupboard. She understood me. Or she was "
	  "detecting floorboard vibrations. She touched both shoulders, both knees, and my neck. I had "
	  "never felt so thoroughly supported.",
	  "I said I was attending a silent retreat. Deborah posted a tapestry of us together. Wove the "
	  "date into it. Tagged the location. My wife arrived before dessert with our wedding album "
	  "and a very specific understanding of 'joint assets.'",
	  "Deborah left me for a moth. Said he had a light about him. My wife got the house, treasure, "
	  "and dungeon. I got this corner. We alternate Christmas with the gravy boat. My warning "
	  "comes with considerable experience and no legal standing."},
	 6},
	{"hair",
	 {"I didn't confess. Her hair did. One snake saw me leaving Deborah's cupboard and told the "
	  "others. By breakfast I was being cross-examined by a fringe.",
	  "I asked to speak to my wife privately. The fringe laughed."},
	 2},
	{"counselling",
	 {"We tried counselling. The counsellor said we needed to stop bringing other people into the "
	  "relationship. Deborah took that personally. She was on the ceiling.",
	  "My wife turned the counsellor to stone. We continued for six weeks. Best listener we ever "
	  "had."},
	 2},
	{"anniversary",
	 {"For our anniversary I bought her shoes. I panicked. The shopkeeper asked what size and I "
	  "said, 'Emotionally, she's quite tall.'",
	  "She bought me a watch. Said I could use it to identify the exact moment I'd ruined the "
	  "evening."},
	 2},
	{"custody",
	 {"We share custody of a basilisk called Martin. Lovely boy. Very sensitive about the divorce. "
	  "Expresses himself mainly through livestock fatalities.",
	  "Every second Sunday I take him to the park. Every second Monday the council sends a "
	  "letter."},
	 2},
	{"letters",
	 {"I wrote her three hundred apology letters. Beautiful things. Honest. Vulnerable. "
	  "Unfortunately, I sealed them with Deborah's silk.",
	  "My wife said that was like apologising for arson on burning stationery. Fair criticism. "
	  "I've grown."},
	 2},
	{"vows",
	 {"I said, 'Till death do us part.' Her hair said, 'Which death?' The celebrant pretended not "
	  "to hear.",
	  "Always clarify the terms when marrying someone whose relatives moult. I am apparently still "
	  "married to one of her previous skins."},
	 2},
	{"dinner",
	 {"I booked a restaurant to win her back. Candlelight. Music. Separate bowls for the hair. I "
	  "thought of everything.",
	  "The waiter asked whether we wanted to split the bill. She said, 'I'd prefer to split him, "
	  "but apparently we're being civil.'"},
	 2},
	{"settlement",
	 {"People think she guards the treasure because she's evil. That treasure is our pension. I'm "
	  "not allowed within thirty paces of it.",
	  "The locked doors were her solicitor's idea. The elaborate puzzles were mine. I was trying "
	  "to remain useful."},
	 2},
	{"vow_of_silence",
	 {"I took a vow of silence after the divorce. Then I discovered you can call almost anything a "
	  "spiritual teaching if you say it slowly beside a wall.",
	  "The beard is not a symbol of wisdom. Deborah has my razor. I refuse to give her the "
	  "satisfaction."},
	 2}};
void monk_dialogue_init(Dialogue *d)
{
	dialogue_init(d, entries, sizeof(entries) / sizeof(entries[0]), 4);
}
