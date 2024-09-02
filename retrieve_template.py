'''
Description: 
Version: 
Author: zrk
Date: 2024-09-01 16:04:48
LastEditors: zrk
LastEditTime: 2024-09-01 19:50:33
'''


template_dict = {"p1": "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.",
                 
                 "p2": "What is the attitude of '{}' toward '{}'? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.", 

                 "p3": "The attitude of '{}' toward '{}' is [MASK]. Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.", 

                 "p4": "TARGET: '{}' TEXT: '{}' Zein da TEXT-ek TARGETekiko duen jarrera? Emaidazu hitz bakarreko erantzuna. Hautatu 'Favor', 'Against' eta 'None'.", 

                 "p5": "Zein jarrera adierazten du '{}' k '{}'ekiko? Emaidazu hitz bakarreko erantzuna. Hautatu 'Favor', 'Against' eta 'None'.",

                 "p6": "'{}'-ren jarrera '{}'-rekin [MASK] da. Emaidazu hitz bakarreko erantzuna. Hautatu 'Favor', 'Against' eta 'None'",

                 "pmix1": "TARGET: '{}' TEXT: '{}' Note that the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. You can consider whether to refer to English supplementary knowledge depending on the situation. What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.",

                 "pmix2": "TARGET: '{}' TEXT: '{}' Note that the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. You can consider whether to refer to English supplementary knowledge depending on the situation. For example, if the Basque sentences show no stance, you can ignore the English supplementary knowledge at all. What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.",

                 "pmix3": "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Please pay more attention on the Basque sentences. If the Basque sentences are not sufficient to determine the stance toward TARGET, you can further refer to English supplementary knowledge. For example, if the Basque sentences show 'None' stance, you can ignore the English supplementary knowledge at all.",

                 "pmix4": "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Please pay more attention on the Basque sentences. If the Basque sentences are not relevant to Vaccine, the answer is 'None'. If the Basque sentences are relevant to Vaccine, please determine the attitude. If the Basque sentences show 'Neutral' stance, you can ignore the English supplementary knowledge at all and output 'None'. If the Basque sentences are not sufficient to determine the stance toward TARGET, you can further refer to English supplementary knowledge.",

                 "pmix5": "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Ignore the English sentences. If the Basque sentences are not relevant to Vaccine, the answer is 'None'.",

                 "pmix6": "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Ignore the English sentences. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine, the answer is 'None'.",

                 "pmix7": "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine, the answer is 'None'.",

                 "pmix8": "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine or express neutral stance toward TARGET, the answer is 'None'. If the Basque sentences are not sufficient to determine the stance toward TARGET, you can further refer to English supplementary knowledge.",

                 "pmix9": "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine or express neutral stance toward TARGET, the answer is 'None'. Ignore the English supplementary knowledge at all.",

                 "pmix10": "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences you can ignore. What is the attitude of Basque sentences toward TARGET? Ignore the English sentences. Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.",


                 "pmix11": "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences you can ignore. What is the attitude of [Basque sentences] toward Vaccine? Ignore every English words in TEXT and focus on the Basque sentences only. Give me a one-word answer. Select from 'Favor', 'Against' and 'None'."


                 
                 
        
                 }

        