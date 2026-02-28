import Mountfuji from '../assets/Mount fuji.jpg'
import LocationTag from '../assets/location-48.png'
export default function Page(props){
    return(
        <>
        <article className='journal-entry' >
                <div className='main-image-container' >
                    <img className="main-image"  src={props.img.src} alt={props.alt} />
                </div>
                <div className='card'>
                    <img className='marker' src={LocationTag} alt="Location icon" />
                    <span>{props.country}</span>
                    <a href={props.googleMapsLink}>View on Google Maps</a>
                    <h2>{props.title}</h2>
                    <p>{props.date}</p>
                    <p>{props.text}</p>
                </div>

        </article>
      
        </>
    )
}