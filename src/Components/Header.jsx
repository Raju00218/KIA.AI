
import Globe from '../assets/globe.png'
export default function Header(){
    return(
        <>
        <header className='App-header'>
                <img className='globe-img' src={Globe} alt="Globe" />
                <h2 className='appName'>Travel World</h2>
        </header>
        </>
    )
}