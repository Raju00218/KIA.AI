import { createClient } from '@supabase/supabase-js'

// Replace these with your actual Supabase project URL and anon key.
// Find these in your Supabase Dashboard under Settings > API.
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'YOUR_SUPABASE_URL'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'YOUR_SUPABASE_ANON_KEY'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
