export const generateContent = async (prompt, mode) => {
    // In a real scenario, this URL would come from your environment variables
    // or a settings input box in the UI. For now, we mock the Colab interaction.
    // Colab URL format is usually: https://<random-string>.ngrok-free.app/generate

    // To use your actual VM backend, replace this with your ngrok URL:
    // Make sure to append the /generate endpoint defined in the FastAPI server.
    const NGROK_URL = "https://yang-monohydrated-pitifully.ngrok-free.dev"; // Paste the URL from your VM terminal here!
    const COLAB_API_URL = `${NGROK_URL}/generate`;

    try {
        if (NGROK_URL && NGROK_URL !== "YOUR_NGROK_URL_HERE") {
            const response = await fetch(COLAB_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true', // Bypasses ngrok's html warning screen
                },
                body: JSON.stringify({ prompt, max_tokens: 150 }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            return { type: mode, content: data.response };
        } else {
            // Mock mode for UI demonstration while Colab is booting up
            await new Promise(resolve => setTimeout(resolve, 2000));

            if (mode === 'text') {
                return {
                    type: 'text',
                    content: `This is a simulated response from the 2B parameter KIA AI model.\n\nYou asked: "${prompt}"\n\nIf the Colab backend was fully connected, you would see the real generated text here.`
                };
            } else {
                return {
                    type: 'image',
                    // Placeholder image for demonstration
                    content: `https://picsum.photos/seed/${encodeURIComponent(prompt)}/512/512`
                };
            }
        }
    } catch (error) {
        console.error("Error communicating with AI backend:", error);
        return {
            type: 'text',
            content: "I'm sorry, I couldn't reach the AI brain on Google Colab. Please ensure the backend is running."
        };
    }
};

export const generateStreamContent = async (prompt, mode, onChunk, options = {}) => {
    // Note: We use the exact base Ngrok URL you provided, targeting the /stream endpoint
    const NGROK_URL = "https://yang-monohydrated-pitifully.ngrok-free.dev"; // Paste the URL from your VM terminal here!
    const COLAB_API_URL = `${NGROK_URL}/stream`;

    // Set default AI parameters if not provided from the UI Settings slider
    const {
        max_tokens = 150,
        temperature = 0.4,       // Lower this! 0.6 is too "wild" for mixed data
        top_k = 40,              // Limits the model to only the most likely words
        repetition_penalty = 1.1 // Lower this! 1.5 is forcing the gibberish
    } = options;

    try {
        if (NGROK_URL && NGROK_URL !== "YOUR_NGROK_URL_HERE") {
            const response = await fetch(COLAB_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true',
                },
                body: JSON.stringify({
                    prompt,
                    max_tokens,
                    temperature,
                    top_k,
                    repetition_penalty
                }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunkText = decoder.decode(value, { stream: true });
                onChunk(chunkText);
            }
        }
    } catch (error) {
        console.error("Error communicating with AI backend stream:", error);
        onChunk("\n[Error: Connection to Colab backend interrupted.]");
    }
};
