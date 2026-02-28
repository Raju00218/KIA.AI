import { supabase } from './supabaseClient';

// Helper to interact with the chat_history table in Supabase

export const loadChatHistory = async () => {
    try {
        const { data, error } = await supabase
            .from('chat_history')
            .select('*')
            .order('created_at', { ascending: false });

        if (error) {
            console.error("Error loading chat history from Supabase:", error);
            return [];
        }

        // Map cloud data format to app state format
        return data.map(item => ({
            id: item.chat_id,
            title: item.title,
            messages: item.messages
        }));
    } catch (err) {
        console.error("Exception loading chat history:", err);
        return [];
    }
};

export const createChat = async (chatId, title, messages) => {
    try {
        const { error } = await supabase
            .from('chat_history')
            .insert([
                { chat_id: chatId, title, messages }
            ]);

        if (error) console.error("Error creating chat in Supabase:", error);
    } catch (err) {
        console.error("Exception creating chat:", err);
    }
};

export const updateChatMessages = async (chatId, messages) => {
    try {
        const { error } = await supabase
            .from('chat_history')
            .update({ messages })
            .eq('chat_id', chatId);

        if (error) console.error("Error updating chat in Supabase:", error);
    } catch (err) {
        console.error("Exception updating chat messages:", err);
    }
};

export const deleteChat = async (chatId) => {
    try {
        const { error } = await supabase
            .from('chat_history')
            .delete()
            .eq('chat_id', chatId);

        if (error) console.error("Error deleting chat from Supabase:", error);
    } catch (err) {
        console.error("Exception deleting chat:", err);
    }
};
