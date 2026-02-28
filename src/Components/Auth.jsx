import React, { useState } from 'react';
import { supabase } from '../services/supabaseClient';
import { Sparkles, X, Mail } from 'lucide-react';
import './Auth.css';

const Auth = ({ onClose }) => {
    const [email, setEmail] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const handleOAuthLogin = async (provider) => {
        const { error } = await supabase.auth.signInWithOAuth({
            provider: provider,
            options: {
                redirectTo: window.location.origin
            }
        });
        if (error) console.error(`Error logging in with ${provider}:`, error.message);
    };

    const handleEmailOTP = async (e) => {
        e.preventDefault();
        if (!email) return;

        setStatusMessage('Sending secure login link...');
        const { error } = await supabase.auth.signInWithOtp({
            email,
            options: {
                shouldCreateUser: true,
                emailRedirectTo: window.location.origin,
            },
        });

        if (error) {
            setStatusMessage(`Error: ${error.message}`);
        } else {
            setStatusMessage('Magic link sent! Check your email inbox to log in.');
            setEmail('');
        }
    };

    return (
        <div className="auth-overlay">
            <div className="auth-card">
                {onClose && (
                    <button className="close-modal-btn" onClick={onClose}>
                        <X size={20} />
                    </button>
                )}

                <div className="auth-logo">
                    <Sparkles size={48} className="text-gradient" />
                </div>
                <h1 className="text-gradient">Sign in to KIA AI</h1>
                <p className="auth-subtitle">Create or access your account to talk to the 2B model and save your chat history.</p>

                <form className="email-login" onSubmit={handleEmailOTP}>
                    <input
                        type="email"
                        placeholder="you@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="email-input"
                        required
                    />
                    <button type="submit" className="auth-btn primary">
                        <Mail size={18} />
                        Continue with Email Link
                    </button>
                </form>

                {statusMessage && (
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                        {statusMessage}
                    </p>
                )}

                <div className="auth-divider">OR</div>

                <div className="auth-buttons">
                    <button onClick={() => handleOAuthLogin('google')} className="auth-btn" type="button">
                        <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" className="provider-icon" />
                        Continue with Google
                    </button>

                    <button onClick={() => handleOAuthLogin('azure')} className="auth-btn" type="button">
                        <img src="https://www.svgrepo.com/show/475666/microsoft-color.svg" alt="Microsoft" className="provider-icon" />
                        Continue with Microsoft
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Auth;
