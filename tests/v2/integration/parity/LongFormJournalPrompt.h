/**
 * @file LongFormJournalPrompt.h
 * @brief Fixed long-form text for initial small-model generation acquisition.
 *
 * A supplied passage gives small instruction models concrete remaining work,
 * unlike an open-ended story or a request for an arbitrary number of tips that
 * they may naturally finish early. This is test input, not an expected answer.
 * The ordinary stochastic sampler still chooses every generated token and EOS
 * remains enabled. Only model definitions explicitly selecting this text use
 * it; topology, KV precision, backend and MTP policy never choose different
 * prompts. Each exact real-model definition owns its fixed acquisition task.
 */
#pragma once

#include "ModelParityGenerationWorkload.h"

namespace llaminar2::test::parity
{
    /** @return Shared immutable source passage; never an expected model output. */
    inline std::string_view longFormJournalText() noexcept
    {
        return
            "At dawn, Mira opened the blue door of the lighthouse and stepped into the harbor. "
            "The fishing boats were still tied to the stone quay. Their ropes creaked softly as the "
            "water rose beneath them. Elias was waiting beside a wooden cart, counting the supplies "
            "they would carry into the mountains. He placed a folded map on a barrel and held its "
            "corners down with four smooth stones. They studied the route together before the "
            "morning wind grew stronger.\n\n"
            "Their first task was to cross the harbor before the market opened. Mira packed bread, "
            "cheese and apples into a canvas bag, while Elias checked the lantern and filled two "
            "bottles with water. A fisherman named Tomas offered to take them across in his small "
            "boat. He showed them where to sit so the weight would remain balanced. As they left "
            "the quay, the lighthouse bell rang three times, and several gulls followed the boat "
            "toward the quieter water beyond the warehouses.\n\n"
            "On the opposite shore, a narrow path ran between vegetable gardens and low stone walls. "
            "The earth was damp from the previous night's rain. Mira noticed small footprints beside "
            "a broken gate and pointed them out to Elias. He thought a fox had passed that way "
            "before sunrise. They stopped at a farmhouse to ask about the upper trail. The farmer "
            "said that a fallen tree blocked the western bridge, but the eastern crossing was open. "
            "She drew a small arrow on their map to show the safer route.\n\n"
            "By late morning, the path entered a forest of tall pines. Sunlight reached the ground "
            "in thin patches, and the air smelled of wet bark. Elias carried the lantern on his "
            "belt, although they did not yet need its light. Mira counted the wooden trail markers "
            "as they passed. At the seventh marker, they found a bench overlooking a stream. They "
            "sat down to eat, spreading a clean cloth over the rough boards. Below them, clear water "
            "moved around the roots of an old tree.\n\n"
            "After their meal, they checked the map again. The observatory stood above the forest, "
            "on a ridge that could not be seen from the harbor. A dotted line marked an old route "
            "used by the people who once worked there. Elias wondered whether the steps would "
            "still be intact. Mira suggested following the newer trail until they reached the next "
            "signpost. They agreed to turn back if the weather changed. Before leaving the bench, "
            "they collected every scrap of paper and secured the food bag inside the larger pack.\n\n"
            "The afternoon brought a colder wind and a view of the distant sea. The trees became "
            "shorter as the trail climbed. Small purple flowers grew between the rocks, and patches "
            "of snow remained in the shaded hollows. Mira stopped beside a spring to refill their "
            "bottles. Elias checked the time and compared the shape of the ridge with the drawing "
            "on the map. A long stone staircase appeared around the next bend. Its lowest steps "
            "were covered with leaves, but a narrow clear line suggested that someone used it regularly.\n\n"
            "Halfway up the staircase, they met a woman carrying a basket of folded blankets. Her "
            "name was Lena, and she looked after the small shelter near the observatory. She told "
            "them that the main gate was locked, but a caretaker would return before evening. "
            "There was a sheltered place to wait beside the southern wall. Lena offered to show "
            "them the path and asked whether they had brought warm coats. Mira thanked her and "
            "helped carry the basket over a section where the stone railing had broken away.\n\n"
            "The shelter had a red roof and two windows facing the valley. Inside, wooden shelves "
            "held cups, candles and notebooks left by earlier visitors. Elias set the lantern on "
            "the table and opened the shutters. Mira placed their coats near the door so they "
            "could find them quickly. Lena showed them a book containing observations of the "
            "weather. Each page recorded the date, wind direction and cloud cover. They read "
            "several entries while waiting for the kettle to warm on the small iron stove.\n\n"
            "Near sunset, footsteps sounded outside the shelter. The caretaker arrived with a ring "
            "of keys and a parcel wrapped in brown paper. He listened as Elias explained where "
            "they had found the map. Then he placed the parcel on the table and carefully removed "
            "its string. Inside was another drawing of the harbor, showing a building that no "
            "longer existed. Mira moved the lantern closer. Together, they compared the two "
            "drawings, tracing the roads and shorelines while the last light faded from the valley.";
    }

    /**
     * @return One immutable English revision task with fixed assistant history.
     * @param followup Model-owned next user request when its template preserves
     *                 the prior token prefix with a non-terminal assistant turn.
     *
     * HTTP closes supplied assistant history; it is not a raw completion prefix.
     * Templates can rewrite non-terminal assistant headers, so the actual token
     * prefix remains a mandatory live proof. The long source passage likewise
     * cannot certify output: every response must produce 384 committed tokens.
     */
    inline ModelParityGenerationPrompt longFormRevisionGenerationPrompt(
        std::optional<std::string> followup = std::nullopt)
    {
        return ModelParityGenerationPrompt(
            "You are an English copy editor. Rewrite the entire supplied passage in clear, detailed English. "
            "Keep every event, character, object and conversation. Include all nine paragraphs in order. "
            "Give the complete revised text, not a summary, explanation, acknowledgement or outline.",
            "Rewrite this entire travel journal in clear English. Include all nine paragraphs with their full detail.\n\n"
                + std::string(longFormJournalText()),
            "I can provide the complete passage with every paragraph in order.", std::move(followup));
    }

    /**
     * @return The existing translation workload for models already proving its horizon.
     *
     * Preserve exact bytes, seed and request geometry for those model identities.
     * This is a separately declared task, not an automatic retry when revision
     * fails. Models selecting it still owe the same strict four-response proof.
     */
    inline ModelParityGenerationPrompt longFormTranslationGenerationPrompt()
    {
        return ModelParityGenerationPrompt(
            "You translate complete English passages into French. Translate every paragraph in order, "
            "preserving all details, names and dialogue. Output only the French translation. "
            "Do not summarize, omit paragraphs, explain the task or add a conclusion.",
            "Translate the following complete travel journal into French. Include all nine paragraphs.\n\n"
                + std::string(longFormJournalText()),
            "The complete French translation of all nine paragraphs follows, in their original order:");
    }
} // namespace llaminar2::test::parity
