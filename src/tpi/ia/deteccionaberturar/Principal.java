package tpi.ia.deteccionaberturar;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;


public class Principal {

	static Mat imagenAberturaOrigen;
	static Mat imagenAberturaTransformada;
	static ArrayList<Point> esquinas;
	static Point esquinaSuperiorIzquierdo;
	static Point esquinaSuperiorDerecho;
	static Point esquinainferiorDerecho;
	static Point esquinainferiorIzquierdo;
	static final double MedidaRadioRealPatron=3.9;
	
	public static void main(String[] args) 
        {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		cargarImagenAberturaOrigen("/prueba1.jpg");
		
		aplicarFiltrosPatronCircular();
		double factorEscala=obtenerFactorDeEscala(MedidaRadioRealPatron);
		
		aplicarFiltrosLineasRectas();
		Mat lineasDetectadas=detectarLineas();
		detectarYDibujarEsquinas(lineasDetectadas);
		calcularDimensionesAbertura(factorEscala);
		
		Imshow im = new Imshow("Imagen PostProcesamiento");
		im.showImage(imagenAberturaOrigen);
	}
	
	
	// carga la imagen en la que se quieren detectar los bordes
	public static void cargarImagenAberturaOrigen(String pathImagen) 
	{
		imagenAberturaOrigen=null;
		URI imagenOrigenUri=null;			
		try 
		{
			imagenOrigenUri = Principal.class.getResource(pathImagen).toURI();
		} 
		catch (URISyntaxException e) {}
		imagenAberturaOrigen = Highgui.imread(new File(imagenOrigenUri).getAbsolutePath());
	}
	
	//aplica los filtros necesarios para detectar las rectas con la Transformada de Hough
	public static void aplicarFiltrosLineasRectas()
	{
		imagenAberturaTransformada=new Mat();
		
		Imgproc.cvtColor( imagenAberturaOrigen, imagenAberturaTransformada, Imgproc.COLOR_RGB2GRAY ); //transforma a escala de grises la imagen origen.
		
		Imgproc.GaussianBlur(imagenAberturaTransformada, imagenAberturaTransformada, new Size(7,7), 10.0, 10.0); //aplica filtro gaussiano.
		
		Imgproc.Canny(imagenAberturaTransformada, imagenAberturaTransformada, 100, 100);
		
	}
	
	//aplica los filtros necesarios para detectar el patr�n circular con la Transformada de Hough
	public static void aplicarFiltrosPatronCircular()
	{
		imagenAberturaTransformada=new Mat();
		
		Imgproc.cvtColor( imagenAberturaOrigen, imagenAberturaTransformada, Imgproc.COLOR_RGB2GRAY ); //transforma a escala de grises la imagen origen.
		
		Imgproc.GaussianBlur(imagenAberturaTransformada, imagenAberturaTransformada, new Size(7,7), 3, 3); //aplica filtro gaussiano.
				
	}
	
	// aplica la Transformada de Hough para detectar las rectas
	public static Mat detectarLineas()
	{
		Mat lineas = new Mat();
		int threshold =50;
		Imgproc.HoughLinesP(imagenAberturaTransformada, lineas, 1, Math.PI/180, threshold,120,8);
		
		//dibuja las lineas rectas que detecta en la imagen original
//		for(int i = 0; i < lineas.cols(); i++) {
//			double[] val = lineas.get(0, i);
//			Core.line(imagenAberturaOrigen, new Point(val[0], val[1]), new Point(val[2], val[3]), new Scalar(0, 0, 255), 2); 
//		}
		return lineas;
	}
		
	//detecta las esquinas de las aberturas y dibuja circulos en su posici�n
	public static void detectarYDibujarEsquinas(Mat lineas) {
		esquinas=new ArrayList<>();
		for (int i = 0; i < lineas.cols(); i++) {
			for (int j = i+1; j < lineas.cols(); j++) {
				Point pt = calcularInterseccionesRectas(lineas.get(0, i), lineas.get(0, j));
				if (pt.x >= 0 && pt.x<=1000 && pt.y >= 0  && pt.y<=1000) {
					esquinas.add(pt);
				}
			}
		}
		System.out.println("Cantidad de Esquinas encontradas="+esquinas.size());
		if(esquinas.size() > 0)
	    {
			for(int i = 0; i < esquinas.size(); i++) {
				Point punto = esquinas.get(i);
	            Core.circle(imagenAberturaOrigen, punto, 5, new Scalar(0, 0, 255), 3);
	        }
	    }
	}
		
	// calcula los puntos de intersecci�n de las rectas encontradas luego de aplicar la Transformada de Hough
	public static Point calcularInterseccionesRectas(double[] a, double[] b) {
		double x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
		float denom = (float) (((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)));
		Point punto = new Point();
		if(denom != 0)
		{
			punto.x = (float) ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2)* (x3 * y4 - y3 * x4))/ denom;
			punto.y = (float) ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2)* (x3 * y4 - y3 * x4))/ denom;
			
			return punto;
		}
		else
			return punto=new Point(-1,-1);
	}
	

	// aplica la Transformada de Hough para detectar el patr�n circular y calcula el factor de escala
	public static double obtenerFactorDeEscala(double medidaRealRadioPatron) {
		Mat circulos = new Mat();
		int radioCirculo=0;

		Imgproc.HoughCircles(imagenAberturaTransformada, circulos,
				Imgproc.CV_HOUGH_GRADIENT,4,
				imagenAberturaTransformada.rows());

		if (circulos.cols() > 0)
			for (int x = 0; x < circulos.cols(); x++) {
				double vCirculos[] = circulos.get(0, x);

				if (vCirculos == null)
					break;

				Point centroCirculo = new Point(Math.round(vCirculos[0]),
						Math.round(vCirculos[1]));
				radioCirculo = (int) Math.round(vCirculos[2]);

				// dibujar patron detectado
				Core.circle(imagenAberturaOrigen, centroCirculo, radioCirculo, new Scalar(0,
						255, 0), 0);
				Core.circle(imagenAberturaOrigen, centroCirculo, 3, new Scalar(0, 255, 0),
						0);
			}
		
		if(radioCirculo!=0)
		{	
			return (float) medidaRealRadioPatron/radioCirculo;  // devuelve el factor de escala
		}
		return 0;		
	}
	
	// calcula la distancia Euclediana entre 2 puntos
	public static double calcularDistanciaEuclideana (Point punto1,Point punto2) {
	    double coordenadaY = Math.abs (punto1.y - punto2.y);
	    double coordenadaX= Math.abs (punto1.x- punto2.x);    
	    double distancia = Math.sqrt((coordenadaY)*(coordenadaY) +(coordenadaX)*(coordenadaX));
	    return distancia; 
	    }	
	
	
	// calcula las dimensiones reales de las aberturas multiplicando la distancia Euclediana por el factor de escala
	public static void calcularDimensionesAbertura(double factorEscala)
	{
		clasificarEsquinas();
		double ladoSuperior= Math.rint(calcularDistanciaEuclideana(esquinaSuperiorIzquierdo, esquinaSuperiorDerecho)*factorEscala*100)/100;
		double ladoInferior=Math.rint(calcularDistanciaEuclideana(esquinainferiorIzquierdo, esquinainferiorDerecho)*factorEscala*100)/100;
		double ladoDerecho=Math.rint(calcularDistanciaEuclideana(esquinaSuperiorDerecho, esquinainferiorDerecho)*factorEscala*100)/100;
		double ladoIzquierdo=Math.rint(calcularDistanciaEuclideana(esquinaSuperiorIzquierdo, esquinainferiorIzquierdo)*factorEscala*100)/100;
		
		System.out.println();
		System.out.println("-----Medidas de la abertura-----");
		System.out.println();
		System.out.println("Lado Superior= "+ladoSuperior+" mm.");
		System.out.println("Lado Inferior= "+ladoInferior+" mm.");
		System.out.println("Lado Derecho= "+ladoDerecho+" mm.");
		System.out.println("Lado Izquierdo= "+ladoIzquierdo+" mm.");
		
	}
	
	// clasifica las esquinas detectadas segun la posicion que ocupan en la abertura
	public static void clasificarEsquinas(){
		
		double sumaMinimo=0;
		double sumaMaximo=0;
		for(int i=0;i<esquinas.size();i++)
		{
			double suma_coordenadas=esquinas.get(i).x+esquinas.get(i).y;
			if(esquinaSuperiorIzquierdo==null|| esquinainferiorDerecho==null)
			{
				esquinaSuperiorIzquierdo=esquinas.get(i);
				esquinainferiorDerecho=esquinas.get(i);
				sumaMinimo=suma_coordenadas;
				sumaMaximo=suma_coordenadas;
				continue;
			}
			if(sumaMinimo>suma_coordenadas)
			{
				sumaMinimo=suma_coordenadas;
				esquinaSuperiorIzquierdo=esquinas.get(i);
				continue;
			}
			if(sumaMaximo<suma_coordenadas)
			{
				sumaMaximo=suma_coordenadas;
				esquinainferiorDerecho=esquinas.get(i);	
				continue;
			}
		}
		esquinas.remove(esquinaSuperiorIzquierdo);
		esquinas.remove(esquinainferiorDerecho);

		if(esquinas.size()==2)
		{
			if (esquinas.get(0).x<esquinas.get(1).x)
			{
				esquinainferiorIzquierdo=esquinas.get(0);
				esquinaSuperiorDerecho=esquinas.get(1);
			}
			else
			{
				esquinainferiorIzquierdo=esquinas.get(1);
				esquinaSuperiorDerecho=esquinas.get(0);
			}
		}	
	}

}
